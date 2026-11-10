import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import difflib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.metrics import fbeta_score, mean_absolute_error, precision_score, recall_score, roc_auc_score

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None

from lightgbm import LGBMClassifier, LGBMRegressor


def resource_path(rel_path: str) -> str:
    """Resolve project-local file paths for dev and packaged execution."""
    if not rel_path:
        return rel_path
    if os.path.isabs(rel_path):
        return rel_path

    rel_path = rel_path.replace("\\", "/")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    candidate = os.path.normpath(os.path.join(project_root, rel_path))
    if getattr(sys, 'frozen', False):
        if rel_path.startswith('src/data/img/'):
            rel_path = rel_path.replace('src/data/img/', 'img/')
        elif rel_path.startswith('src/data/dataset/'):
            rel_path = rel_path.replace('src/data/dataset/', 'dataset/')
        elif rel_path.startswith('src/data/database/'):
            rel_path = rel_path.replace('src/data/database/', 'database/')
        candidate = os.path.join(sys._MEIPASS, rel_path)  # type: ignore[attr-defined]
    return candidate


def _excel_sha1(path: str) -> str:
    """Fast SHA1 hash for cache validation."""
    digest = hashlib.sha1()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _model_path_for_excel(excel_filenames: tuple[str, ...]) -> str:
    """Where to store the cached model bundle."""
    base_dir = Path(os.path.dirname(__file__)) / 'models'
    base_dir.mkdir(exist_ok=True)
    stem = '_'.join(Path(filename).stem for filename in excel_filenames)
    return str(base_dir / f'{stem}_lgbm.joblib')


def _combined_excel_sha1(paths: List[str]) -> str:
    """Create a single cache signature for multiple input workbooks."""
    digest = hashlib.sha1()
    for path in paths:
        digest.update(os.path.basename(path).encode('utf-8'))
        digest.update(_excel_sha1(path).encode('ascii'))
    return digest.hexdigest()


class AIModel:
    """
    LightGBM-based credit scoring model with the newer preprocessing pipeline.
    Keeps the app-facing API compatible with the workflow already used by the UI.
    """

    TARGET = 'Longest DPD'

    CATEGORICAL_COLS: List[str] = [
        'Asset',
        'Manufacturer',
        'Type of Unit',
        'Loan History',
        'BAP Type',
        'Employment Type',
        'Home Ownership',
        'Citizenship',
        'Dealer Name',
        'Submitted ID',
        'Product Type',
        'Mode of Payment',
        'Co-Borrowers',
        'Borrower Type',
        'Marital Status',
    ]

    HARISH_TE_COLS = [
        'Dealer Name', 'Manufacturer', 'Asset', 'Employment Type',
        'Product Type', 'Home Ownership', 'Loan History', 'BAP Type',
        'Mode of Payment',
    ]

    RAW_INPUT_FEATURES = [
        'Amount Finance', 'Downpayment', 'Downpayment %', 'Payment Terms',
        'Asset', 'Manufacturer', 'Type of Unit', 'Price Gap', 'Loan History',
        'BAP Type', 'BAP Amount', 'Interest Rate', 'Claimed Income',
        'EV Income', 'BV Income', 'Employment Type', 'Claimed Tenure',
        'EV Tenure', 'BV Tenure', 'DSCR', 'Age as of Contract',
        'Home Ownership', 'Residence Tenure', 'Citizenship', 'Dealer Name',
        'Submitted ID', 'Marital Status', 'Loan Amort', 'Product Type',
        'Mode of Payment', 'Co-Borrowers', 'Borrower Type', 'Contract Start Date',
    ]

    HARISH_DROP_COLS = {
        'Citizenship', 'Contract_Year', 'Type of Unit', 'Borrower Type',
        'Contract ID',
        'Dealer_Default_Rate', 'LoanHistory_x_DealerRisk', 'Dealer_x_LowDP',
        'Dealer_x_Term', 'High_Risk_Dealer', 'Mid_Risk_Dealer',
        'Dealer_Risk_Tier', 'GoodBorrower_RiskyDealer', 'Submitted ID',
    }

    HARISH_BEST_PARAMS = {
        'n_estimators': 1386,
        'max_depth': 12,
        'num_leaves': 184,
        'min_child_samples': 6,
        'learning_rate': 0.1538862143578627,
        'subsample': 0.7517993141753957,
        'colsample_bytree': 0.6008938631093506,
        'reg_alpha': 0.00012940233398199124,
        'reg_lambda': 0.00022914020156036827,
        'scale_pos_weight': 4.0,  # ~80% neg / ~20% pos for DPD>0 target (any DPD detection)
    }

    MODEL_PARAMETER_VARIANTS = (
        {},
        {'scale_pos_weight': 2.0},
        {'scale_pos_weight': 8.0},
        {
            'n_estimators': 1600,
            'max_depth': 8,
            'num_leaves': 63,
            'min_child_samples': 20,
            'learning_rate': 0.05,
        },
    )

    MODEL_VERSION = 20
    DPD_CATEGORY_LABELS = (
        'Current',
        'Early Delinquency',
        'Moderate Delinquency',
        'Serious Delinquency',
        'Severe Delinquency',
    )
    MIN_VALIDATION_PRECISION = 0.75

    @staticmethod
    def make_balanced_dataset(df, target, random_seed):
        df_pos = df[df[target] != 0]
        df_neg = df[df[target] == 0]
        min_size = min(len(df_pos), len(df_neg))
        df_pos_sampled = df_pos.sample(n=min_size, random_state=random_seed)
        df_neg_sampled = df_neg.sample(n=min_size, random_state=random_seed)
        return pd.concat([df_pos_sampled, df_neg_sampled]).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # @staticmethod
    # def quantize_longest_dpd(x):
    #     try:
    #         return 0 if x <= 30 else 1
    #     except Exception:
    #         return np.nan

    def __init__(self, *excel_filenames: str):
        if not excel_filenames:
            raise ValueError('AIModel requires at least one Excel training workbook.')

        self.excel_paths = [resource_path(filename) for filename in excel_filenames]
        missing = [path for path in self.excel_paths if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError('Training workbook(s) not found: ' + ', '.join(missing))

        self.model_path = _model_path_for_excel(tuple(excel_filenames))
        self._loaded_from_cache = False
        self._ext_enc = None
        self._explainer = None

        excel_sig = _combined_excel_sha1(self.excel_paths)
        if os.path.exists(self.model_path):
            try:
                bundle = joblib.load(self.model_path)
                if bundle.get('version') == self.MODEL_VERSION and bundle.get('excel_sha1') == excel_sig:
                    self._load_bundle(bundle)
                    self._ext_enc = self._try_load_external_encoding()
                    self._loaded_from_cache = True
                    print(f'[AIModel] Loaded cached model: {self.model_path}')
                    return
                print('[AIModel] Cache exists but signature/version mismatch. Retraining…')
            except Exception as exc:
                print(f'[AIModel] Failed to load cache ({exc}). Retraining…')

        training_data = self.process_data(self._load_training_data())
        self._train_and_cache(training_data, excel_sig)
        self._ext_enc = self._try_load_external_encoding()

    def _load_training_data(self) -> pd.DataFrame:
        """Load one or two workbooks and align them for training."""
        if len(self.excel_paths) == 1:
            data = pd.read_excel(self.excel_paths[0])
            return data.drop_duplicates().reset_index(drop=True)

        original, updated = [pd.read_excel(path) for path in self.excel_paths]
        updated = updated.rename(columns={'Unit Brand': 'Manufacturer'})
        original = original.drop(columns=['Product Type.1'], errors='ignore')
        updated = updated.drop(columns=[
            'Address', 'Claimed Employer', 'Claimed Employer Type',
            'Claimed Position', 'Credit Info ID', 'EV Employer',
            'EV Employer Type', 'EV Position', 'Gender', 'No of Dependents',
        ], errors='ignore')
        if 'Borrower Type' not in original:
            original['Borrower Type'] = 'Principal'
        shared_columns = [column for column in original.columns if column in updated.columns]
        if not shared_columns:
            raise ValueError('The training workbooks have no shared columns.')
        combined = pd.concat([original[shared_columns], updated[shared_columns]], ignore_index=True)
        combined = combined.drop(columns=['Party No', 'Application No'], errors='ignore')
        return combined.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _clean_income(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).lower().strip()
        if any(word in text for word in ('confidential', 'not disclosed', 'not provided', 'proof', 'coec', 'pls refer', 'cannot disclose')):
            return np.nan
        text = text.replace(',', '')
        text = re.sub(r'(\d+)k', r'\g<1>000', text)
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if len(numbers) == 1:
            amount = float(numbers[0])
            return amount / 12 if 'annual' in text else amount
        if len(numbers) >= 2:
            return float(np.mean([float(numbers[0]), float(numbers[1])]))
        return np.nan

    @staticmethod
    def _clean_tenure(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        text = str(value).lower().strip().replace('?', '')
        if re.fullmatch(r'\d+(?:\.\d+)?', text):
            return float(text)
        years = re.search(r'(\d+)\s*(?:year|years|yr|yrs)', text)
        months = re.search(r'(\d+)\s*(?:month|months|mos)', text)
        if years or months:
            return float((int(years.group(1)) if years else 0) * 12 + (int(months.group(1)) if months else 0))
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return float(numbers[0]) if numbers else np.nan

    @classmethod
    def process_data(cls, data: pd.DataFrame, inference: bool = False) -> pd.DataFrame:
        """Apply the notebook's cleaning and feature engineering."""
        data = data.copy().rename(columns={'Current DPD': 'Longest_DPD'})

        numeric_columns = [
            'Amount Finance', 'Downpayment', 'Downpayment %', 'Payment Terms',
            'BAP Amount', 'Interest Rate', 'Claimed Income', 'EV Income', 'BV Income',
            'Claimed Tenure', 'EV Tenure', 'BV Tenure', 'DSCR', 'Age as of Contract',
            'Residence Tenure', 'Loan Amort', 'Price Gap',
            'Contract_Year', 'Contract_Month'
        ]
        for column in numeric_columns:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors='coerce')

        for column in ('Claimed Income', 'EV Income', 'BV Income'):
            if column in data:
                data[column] = data[column].apply(cls._clean_income)
        for column in ('Claimed Tenure', 'EV Tenure', 'BV Tenure', 'Residence Tenure'):
            if column in data:
                data[column] = data[column].apply(cls._clean_tenure)
        for column in ('BAP Amount', 'EV Income', 'EV Tenure', 'BV Income', 'Price Gap', 'BV Tenure', 'DSCR', 'Claimed Tenure', 'Claimed Income', 'Residence Tenure'):
            if column in data:
                data[f'{column}_missing'] = data[column].isna().astype(int)
                data[column] = pd.to_numeric(data[column], errors='coerce').fillna(-999)
        for column in cls.CATEGORICAL_COLS:
            if column in data:
                data[column] = data[column].fillna('Unknown')
            else:
                data[column] = 'Unknown'
        if 'Contract Start Date' in data:
            dates = pd.to_datetime(data['Contract Start Date'], errors='coerce')
            data['Contract_Year'], data['Contract_Month'] = dates.dt.year, dates.dt.month
            data = data.drop(columns=['Contract Start Date'])
        if 'Marital Status' in data:
            data['Marital Status'] = data['Marital Status'].replace({'Separated': 'Without Partner', 'Divorced': 'Without Partner', 'Single': 'Single/Not Married', 'Not Married': 'Single/Not Married'})
            if not inference:
                data = data[~data['Marital Status'].isin(['Partner', 'Unspecified'])]
        data = data.drop(columns=['Birthday', 'BAP Date'], errors='ignore')
        if 'Longest_DPD' in data:
            data['Longest_DPD'] = pd.to_numeric(data['Longest_DPD'], errors='coerce').fillna(0)
            data['Target'] = (data['Longest_DPD'] > 1).astype(int)  # DPD > 1 is the default definition

        numeric_columns = ['BAP Amount', 'EV Income', 'EV Tenure', 'BV Income', 'Price Gap', 'BV Tenure', 'DSCR', 'Claimed Tenure', 'Claimed Income', 'Residence Tenure', 'Amount Finance', 'Downpayment', 'Payment Terms', 'Interest Rate', 'Loan Amort', 'Age as of Contract', 'Downpayment %']
        for column in numeric_columns:
            if column not in data:
                data[column] = np.nan
            if f'{column}_missing' not in data:
                data[f'{column}_missing'] = 1
        for column in ('BAP Type', 'Mode of Payment'):
            if column not in data:
                data[column] = 'Unknown'
        numeric_columns = data.select_dtypes(include=['number']).columns
        data[numeric_columns] = data[numeric_columns].replace(-999, np.nan)
        epsilon = 1e-9

        def divide(left, right):
            if left not in data or right not in data:
                return pd.Series(np.nan, index=data.index)
            return data[left] / data[right].replace(0, np.nan)

        data['DTI'] = divide('Loan Amort', 'Claimed Income')
        data['Income_Gap'] = data['Claimed Income'] - data['BV Income']
        data['LTV'] = data['Amount Finance'] / (data['Amount Finance'] + data['Downpayment']).replace(0, np.nan)
        data['Repayment_Stress'] = divide('Loan Amort', 'Amount Finance')
        data['BAP_to_Finance'] = divide('BAP Amount', 'Amount Finance')
        data['BAP_Income_Risk'] = ((data['BAP Amount'] > 200000) & (data['Income_Gap'] < 0)).astype(int)
        data['BAP_Asset_Risk'] = ((data['BAP Amount'] > 200000) & (data['Price Gap'] < -50000)).astype(int)
        data['Collateral_Shortfall'] = (-data['Price Gap']).clip(lower=0) / (data['Amount Finance'] + epsilon)
        data['Rate_x_Term'] = data['Interest Rate'] * data['Payment Terms']
        data['Equity_Cushion'] = data['Downpayment'] / (data['Downpayment'] + data['Amount Finance'] + epsilon)
        data['Shortfall_x_Term'] = data['Collateral_Shortfall'] * data['Payment Terms']
        data['Income_Gap_x_LTV'] = data['Income_Gap'] * data['LTV']
        data['DTI_x_LTV'] = data['DTI'] * data['LTV']
        data['DP_to_Income'] = divide('Downpayment', 'Claimed Income')
        data['Age_at_Maturity'] = data['Age as of Contract'] + data['Payment Terms'] / 12
        data['High_DTI'] = (data['DTI'] > 0.40).astype(int)
        data['Tenure_Mismatch'] = (data['Claimed Tenure'] - data['BV Tenure']).abs()
        data['Income_Inconsistency'] = (data['BV Income'] / (data['Claimed Income'] + epsilon)).clip(upper=5)
        verification_columns = ['EV Income', 'BV Income', 'EV Tenure', 'BV Tenure', 'Price Gap', 'DSCR', 'Claimed Income', 'Claimed Tenure']
        data['Verification_Gaps'] = sum(data[column].isna().astype(int) for column in verification_columns if column in data)
        data['High_Risk_Missing'] = ((data.get('BV Tenure_missing', 0) == 1) & (data.get('Claimed Tenure_missing', 0) == 1)).astype(int)
        data['Unverified_Income_AND_Tenure'] = ((data.get('BV Income_missing', 0) == 1) & (data.get('Claimed Tenure_missing', 0) == 1)).astype(int)
        data['Long_Term_Loan'] = (data['Payment Terms'] >= 37).astype(int)
        downpayment_percent = data['Downpayment %']
        data['Danger_DP_Band'] = ((downpayment_percent >= 15) & (downpayment_percent < 20)).astype(int)
        data['Safe_DP'] = (downpayment_percent >= 30).astype(int)
        data['LongTerm_x_DangerDP'] = ((data['Payment Terms'] >= 37) & (downpayment_percent >= 15) & (downpayment_percent < 20)).astype(int)
        data['BAP_Adverse_Loans_Only'] = (data['BAP Type'] == 'Adverse Loans').astype(int)
        mop_risk = {
            'OTC - Bank': 4, 'OTC - Gcash (Merchant)': 3, 'ADA - Security': 3,
            'OTC': 3, 'ADA - BPI': 2, 'ADA - RCBC': 2, 'ADA - BDO': 2,
            'ADA - Metrobank': 1, 'PDC': 1, 'ADA - Eastwest': 0,
            'Salary Deduction': 0, 'OTC - Gcash (Bank Transfer)': 0,
        }
        data['MOP_Risk_Score'] = data['Mode of Payment'].map(mop_risk).fillna(1)
        data['High_Risk_MOP'] = data['Mode of Payment'].isin(['OTC', 'ADA - Security', 'OTC - Gcash (Merchant)']).astype(int)
        loan_history_risk = {
            'With history of Pastdue': 3, 'Poor Handling': 3,
            'Loans and other credit dealings seen on CIC': 2,
            'Existing client of JFP less than 25% Paid': 2,
            'Repeat clients of other Banks and Finances': 1,
            'Existing client of JFP 25% and above Paid': 1,
            'Existing client of JFP 50% and above Paid': 0,
            'Fully paid Good Repeat Client': 0,
        }
        data['Loan_History_Risk'] = data['Loan History'].map(loan_history_risk).fillna(1)
        data['HighRiskMOP_x_DealerTier'] = np.nan

        # --- New risk features (fix for Personal Loans / zero-collateral) ---
        if 'Product Type' in data:
            data['Is_Personal_Loan'] = data['Product Type'].str.contains('Personal', case=False, na=False).astype(int)
        else:
            data['Is_Personal_Loan'] = 0
        bv_missing = data.get('BV Income_missing', pd.Series(1, index=data.index))
        data['Zero_DP_AND_No_BV'] = ((downpayment_percent == 0) & (bv_missing == 1)).astype(int)
        ev_inc_m = data.get('EV Income_missing', pd.Series(1, index=data.index))
        bv_inc_m = data.get('BV Income_missing', pd.Series(1, index=data.index))
        ev_ten_m = data.get('EV Tenure_missing', pd.Series(1, index=data.index))
        bv_ten_m = data.get('BV Tenure_missing', pd.Series(1, index=data.index))
        data['Full_Verification_Missing'] = ((ev_inc_m == 1) & (bv_inc_m == 1) & (ev_ten_m == 1) & (bv_ten_m == 1)).astype(int)
        data['Collateral_Free_Loan'] = (data['Is_Personal_Loan'] & data['Zero_DP_AND_No_BV']).astype(int)
        data['FullMissing_x_HighLTV'] = data['Full_Verification_Missing'] * data['LTV'].fillna(1.0)
        data['Personal_Loan_Risk_Index'] = (
            data['Is_Personal_Loan'] * 3 +
            data['Full_Verification_Missing'] * 2 +
            data['Zero_DP_AND_No_BV'] * 2
        )
        return data.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    def process_raw_input(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the same cleaning to a single raw row."""
        processed = self.process_data(pd.DataFrame([row_dict]))
        if processed.empty:
            raise ValueError('Customer input produced no row after preprocessing.')
        return processed.drop(columns=['Longest_DPD', 'Target'], errors='ignore').iloc[0].to_dict()

    def _try_load_external_encoding(self):
        """Load optional encoding maps for custom labels."""
        try:
            enc_path = resource_path(os.path.join('encoding_report', 'encoding_maps.json'))
            if not os.path.exists(enc_path):
                return None
            with open(enc_path, 'r', encoding='utf-8') as handle:
                enc = json.load(handle)
            cat_maps = enc['categorical_maps']
            fallback_code = enc['fallback_code']
            features_order = enc['features_order']
            categorical_cols = list(cat_maps.keys())
            numeric_cols = [column for column in features_order if column not in categorical_cols]
            lower_vocab = {column: {key.lower(): key for key in mapping.keys()} for column, mapping in cat_maps.items()}

            builtin_aliases = {
                'Citizenship  - type': {'filipino': 'Philippines', 'philippines': 'Philippines', 'ph': 'Philippines', 'pinoy': 'Philippines'},
                'Product Type': {'used': 'Auto Loan (Second Hand)', 'second hand': 'Auto Loan (Second Hand)', 'second-hand': 'Auto Loan (Second Hand)'},
                'Submitted ID   - type': {'primary': 'ID2', 'id2': 'ID2'},
                'Co-Borrowers': {'yes': 'Yes', 'no': 'No', '1': '1', '0': '0'},
                'Mode of Payment': {'pdc': 'Post Dated Checks', 'post dated checks': 'Post Dated Checks', 'post-dated checks': 'Post Dated Checks', 'auto-debit': 'Auto Debit Arrangement'},
                'Employment Type  - type': {'locally employed': 'Locally employed', 'self-employed': 'Self-Employed', 'self employed': 'Self-Employed', 'self employed and employed': 'Self-Employed and Employed'},
                'Home Ownership - TYPE': {'owned': 'Owned', 'renting': 'Renting', 'rent': 'Renting'},
                'LRD': {'low': 'Low', 'normal': 'Normal', 'high': 'High'},
            }

            def load_user_aliases(csv_rel='encoding_report/aliases.csv'):
                try:
                    csv_path = resource_path(csv_rel)
                    if not os.path.exists(csv_path):
                        return {}
                    frame = pd.read_csv(csv_path)
                    aliases = {}
                    for _, row in frame.iterrows():
                        col = str(row['column']).strip()
                        alias = str(row['alias']).strip()
                        canon = str(row['canonical']).strip()
                        aliases.setdefault(col, {})[alias.lower()] = canon
                    return aliases
                except Exception:
                    return {}

            user_aliases = load_user_aliases()
            return {
                'cat_maps': cat_maps,
                'fallback_code': fallback_code,
                'features_order': features_order,
                'categorical_cols': categorical_cols,
                'numeric_cols': numeric_cols,
                'lower_vocab': lower_vocab,
                'builtin_aliases': builtin_aliases,
                'user_aliases': user_aliases,
            }
        except Exception as exc:
            print(f'[AIModel] Failed to load external encoding maps: {exc}')
            return None

    def _alias_lookup(self, col: str, sval: str) -> Optional[str]:
        if not self._ext_enc:
            return None
        low = sval.lower().strip()
        ua = self._ext_enc['user_aliases']
        ba = self._ext_enc['builtin_aliases']
        if col in ua and low in ua[col]:
            return ua[col][low]
        if col in ba and low in ba[col]:
            return ba[col][low]
        return None

    def _normalize_label(self, col: str, val) -> str:
        if val is None or (isinstance(val, str) and val.strip() == ''):
            return 'Missing'
        sval = str(val).strip()
        if self._ext_enc and col in self._ext_enc['cat_maps'] and isinstance(val, (int, float, np.integer, np.floating)):
            if float(val).is_integer():
                key = str(int(val))
                if key in self._ext_enc['cat_maps'][col]:
                    return key
        canon = self._alias_lookup(col, sval)
        if canon:
            return canon
        if self._ext_enc and col in self._ext_enc['lower_vocab']:
            low = sval.lower()
            if low in self._ext_enc['lower_vocab'][col]:
                return self._ext_enc['lower_vocab'][col][low]
        return sval

    def _fuzzy_match_label(self, col: str, sval: str, cutoff=0.92) -> Optional[str]:
        if not self._ext_enc or col not in self._ext_enc['cat_maps']:
            return None
        choices = list(self._ext_enc['cat_maps'][col].keys())
        matches = difflib.get_close_matches(sval, choices, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def _map_categorical_to_code_ext(self, col: str, val, notes: list) -> int:
        mapping = self._ext_enc['cat_maps'][col]
        fallback = int(self._ext_enc['fallback_code'][col])
        valid_codes = set(mapping.values())
        if isinstance(val, (int, np.integer)) and int(val) in valid_codes:
            return int(val)
        if isinstance(val, (float, np.floating)) and val.is_integer() and int(val) in valid_codes:
            return int(val)
        if isinstance(val, str) and val.isdigit() and int(val) in valid_codes:
            return int(val)
        cand = self._normalize_label(col, val)
        if cand in mapping:
            return mapping[cand]
        close = self._fuzzy_match_label(col, cand, cutoff=0.92)
        if close and close in mapping:
            if cand != close:
                notes.append(f'[fuzzy:{col}] {cand!r} -> {close!r}')
            return mapping[close]
        notes.append(f'[fallback:{col}] {val!r} -> code {fallback}')
        return fallback

    def _normalize_numeric_ext(self, col: str, val, notes: list) -> float:
        if col == 'Contract Start Date':
            dt = pd.to_datetime(val, format='%m/%d/%Y', errors='coerce')
            if pd.isna(dt):
                dt = pd.to_datetime(val, errors='coerce')
            return 0.0 if pd.isna(dt) else float(dt.value)
        if col == 'DSCR - %':
            num = pd.to_numeric(val, errors='coerce')
            if pd.isna(num):
                return 0.0
            x = float(num)
            if 1.5 < x <= 200:
                notes.append(f'[scale:{col}] {x} -> {x / 100.0}')
                return x / 100.0
            return x
        num = pd.to_numeric(val, errors='coerce')
        return float(0.0 if pd.isna(num) else num)

    def _encode_with_external(self, row_dict: Dict[str, Any], verbose=False) -> pd.DataFrame:
        notes, row = [], {}
        feat_order = self._ext_enc['features_order']
        cat_cols = set(self._ext_enc['categorical_cols'])
        for column in feat_order:
            val = row_dict.get(column, None)
            if column in cat_cols:
                row[column] = self._map_categorical_to_code_ext(column, val, notes)
            else:
                row[column] = self._normalize_numeric_ext(column, val, notes)
        if verbose and notes:
            print('\n'.join(notes))
        frame = pd.DataFrame([row])
        return frame.reindex(columns=self.features, fill_value=0)

    def _fit_harish_splits(self, train_data, val_data, test_data, train_target):
        from sklearn.model_selection import KFold

        global_mean = float(train_target.mean())
        train = train_data.drop(columns=['Longest_DPD', 'Target'], errors='ignore').copy()
        val = val_data.drop(columns=['Longest_DPD', 'Target'], errors='ignore').copy()
        test = test_data.drop(columns=['Longest_DPD', 'Target'], errors='ignore').copy()
        winsor_columns = (
            'Loan Amort', 'Claimed Income', 'Downpayment', 'Amount Finance',
            'Residence Tenure', 'BV Income', 'EV Income', 'BAP Amount',
            'Claimed Tenure', 'Interest Rate', 'Price Gap', 'DSCR',
        )
        self.winsor_limits = {}
        for column in winsor_columns:
            if column not in train:
                continue
            values = pd.to_numeric(train[column], errors='coerce')
            lower, upper = values.quantile([0.01, 0.99]).tolist()
            if pd.notna(lower) and pd.notna(upper):
                self.winsor_limits[column] = (float(lower), float(upper))
        for part in (train, val, test):
            self._apply_winsor_limits(part, self.winsor_limits)
        drop_high_card = ['Dealer Name', 'Asset', 'Manufacturer', 'Loan History', 'BAP Type']

        dealer = 'Dealer Name'
        if dealer in train:
            stats = pd.DataFrame({'target': train_target.values, 'dealer': train[dealer].astype(str).values}).groupby('dealer')['target'].agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + global_mean * 10) / (stats['count'] + 10)
            rates = []
            for part in (train, val, test):
                rates.append(part[dealer].astype(str).map(smooth).fillna(global_mean))
            for part, rate in zip((train, val, test), rates):
                part['Dealer_Default_Rate'] = rate.to_numpy()
                part['LoanHistory_x_DealerRisk'] = part['Loan_History_Risk'] * rate.to_numpy()
                part['Dealer_x_LowDP'] = rate.to_numpy() * (1 - part['Downpayment'] / 100)
                part['Dealer_x_Term'] = rate.to_numpy() * part['Payment Terms']
            high, mid = smooth.quantile(.90), smooth.quantile(.70)
            self.dealer_smooth = smooth.to_dict()
            self.dealer_high_cutoff = float(high)
            self.dealer_mid_cutoff = float(mid)
            self.dealer_dti_cutoff = float(train['DTI'].quantile(.30))
            for part, rate in zip((train, val, test), rates):
                tier = pd.cut(rate, [-np.inf, mid, high, np.inf], labels=[0, 1, 2]).astype(int)
                part['Dealer_Risk_Tier'] = tier.to_numpy()
                part['High_Risk_Dealer'] = (tier == 2).astype(int).to_numpy()
                part['Mid_Risk_Dealer'] = (tier == 1).astype(int).to_numpy()
                part['GoodBorrower_RiskyDealer'] = ((tier >= 1) & (part['DTI'] < train['DTI'].quantile(.30))).astype(int).to_numpy()
                part['HighRiskMOP_x_DealerTier'] = part['High_Risk_MOP'] * tier.to_numpy()

        target_maps = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for column in self.HARISH_TE_COLS:
            if column not in train:
                continue
            grouped = pd.DataFrame({'target': train_target.values, 'value': train[column].astype(str).values}).groupby('value')['target'].agg(['mean', 'count'])
            mapping = ((grouped['mean'] * grouped['count'] + global_mean * 20) / (grouped['count'] + 20)).to_dict()
            target_maps[column] = mapping
            oof = np.full(len(train), global_mean, dtype=float)
            values = train[column].astype(str).reset_index(drop=True)
            targets = train_target.reset_index(drop=True)
            for fit_idx, hold_idx in kfold.split(values):
                fold = pd.DataFrame({'target': targets.iloc[fit_idx].values, 'value': values.iloc[fit_idx].values}).groupby('value')['target'].agg(['mean', 'count'])
                fold_map = ((fold['mean'] * fold['count'] + global_mean * 20) / (fold['count'] + 20)).to_dict()
                oof[hold_idx] = values.iloc[hold_idx].map(fold_map).fillna(global_mean)
            train[f'{column}_TE'] = oof
            val[f'{column}_TE'] = val[column].astype(str).map(mapping).fillna(global_mean).to_numpy()
            test[f'{column}_TE'] = test[column].astype(str).map(mapping).fillna(global_mean).to_numpy()

        dealer_features = {'Dealer_Default_Rate', 'LoanHistory_x_DealerRisk', 'Dealer_x_LowDP', 'Dealer_x_Term', 'High_Risk_Dealer', 'Mid_Risk_Dealer', 'Dealer_Risk_Tier', 'GoodBorrower_RiskyDealer', 'HighRiskMOP_x_DealerTier'}
        drop_columns = set(drop_high_card) | (self.HARISH_DROP_COLS - dealer_features)
        for part in (train, val, test):
            part.drop(columns=[c for c in drop_columns if c in part], inplace=True)
        categorical = [c for c in self.CATEGORICAL_COLS if c in train.columns and c not in drop_high_card]
        category_maps = {}
        for column in categorical:
            values = pd.concat([train[column], val[column], test[column]]).fillna('Unknown').astype(str)
            mapping = {value: code for code, value in enumerate(sorted(values.unique()))}
            category_maps[column] = mapping
            for part in (train, val, test):
                part[column] = part[column].fillna('Unknown').astype(str).map(mapping).fillna(-1)

        train = train.apply(pd.to_numeric, errors='coerce')
        val = val.apply(pd.to_numeric, errors='coerce')
        test = test.apply(pd.to_numeric, errors='coerce')
        impute_columns = [c for c in ('Claimed Income', 'Claimed Tenure', 'DSCR', 'Residence Tenure') if c in train]
        if impute_columns:
            imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1), max_iter=5, random_state=42)
            train[impute_columns] = imputer.fit_transform(train[impute_columns])
            val[impute_columns] = imputer.transform(val[impute_columns])
            test[impute_columns] = imputer.transform(test[impute_columns])
        else:
            imputer = None
        medians = train.median().fillna(0)
        train, val, test = train.fillna(medians), val.fillna(medians), test.fillna(medians)

        scores, thresholds = {}, {}
        for column in train.columns:
            values = train[column].to_numpy(float)
            cuts = np.percentile(values, np.linspace(5, 95, 50))
            best_score, best_cut = 0.0, None
            for cut in np.unique(cuts):
                left, right = train_target.to_numpy()[values <= cut], train_target.to_numpy()[values > cut]
                if len(left) < 5 or len(right) < 5:
                    continue
                pooled = np.sqrt((left.var() + right.var()) / 2)
                score = abs(left.mean() - right.mean()) / pooled if pooled else 0
                if score > best_score:
                    best_score, best_cut = score, cut
            scores[column] = best_score
            if best_score > 0.5:
                thresholds[column] = float(best_cut)
        for column, threshold in thresholds.items():
            name = f'{column}_cd_split'
            train[name] = (train[column] <= threshold).astype(int)
            val[name] = (val[column] <= threshold).astype(int)
            test[name] = (test[column] <= threshold).astype(int)
        redundant = ['LoanHistory_x_DealerRisk_cd_split', 'Dealer_x_LowDP_cd_split', 'Dealer_Default_Rate_cd_split', 'Dealer_x_Term_cd_split', 'GoodBorrower_RiskyDealer_cd_split', 'High_Risk_Dealer_cd_split', 'HighRiskMOP_x_DealerTier_cd_split', 'High_DTI', 'GoodBorrower_RiskyDealer']
        train.drop(columns=[c for c in redundant if c in train], inplace=True)
        val.drop(columns=[c for c in redundant if c in val], inplace=True)
        test.drop(columns=[c for c in redundant if c in test], inplace=True)
        features = sorted(train.columns, key=lambda column: scores.get(column, 0), reverse=True)
        return train[features], val.reindex(columns=features, fill_value=0), test.reindex(columns=features, fill_value=0), target_maps, category_maps, global_mean, medians, thresholds, imputer if impute_columns else None

    def _fit_harish_features(self, data: pd.DataFrame):
        target = data['Target'].astype(int)
        features = data.drop(columns=['Longest_DPD', 'Target'], errors='ignore').copy()
        features = features.drop(columns=[c for c in self.HARISH_DROP_COLS if c in features], errors='ignore')
        categorical = [c for c in self.CATEGORICAL_COLS if c in features.columns]
        global_mean = float(target.mean())
        target_maps = {}
        for column in self.HARISH_TE_COLS:
            if column not in features.columns:
                continue
            grouped = target.groupby(features[column].astype(str)).agg(['mean', 'count'])
            target_maps[column] = {str(key): float((row['mean'] * row['count'] + global_mean * 20) / (row['count'] + 20)) for key, row in grouped.iterrows()}
            features[f'{column}_TE'] = features[column].astype(str).map(target_maps[column]).fillna(global_mean)
        category_maps = {}
        for column in categorical:
            values = features[column].fillna('Unknown').astype(str)
            mapping = {value: code for code, value in enumerate(sorted(values.unique()))}
            category_maps[column] = mapping
            features[column] = values.map(mapping).astype(float)
        features = features.apply(pd.to_numeric, errors='coerce')
        medians = features.median().fillna(0.0)
        features = features.fillna(medians)
        split_thresholds = {}
        for column in features.columns:
            values = features[column].to_numpy(dtype=float)
            thresholds = np.percentile(values, np.linspace(5, 95, 50))
            best_score, best_threshold = 0.0, None
            for threshold in np.unique(thresholds):
                left = target[features[column] <= threshold]
                right = target[features[column] > threshold]
                if len(left) < 5 or len(right) < 5:
                    continue
                pooled = np.sqrt((left.var() + right.var()) / 2)
                score = abs(left.mean() - right.mean()) / pooled if pooled > 0 else 0.0
                if score > best_score:
                    best_score, best_threshold = float(score), float(threshold)
            if best_score > 0.5:
                split_thresholds[column] = best_threshold
                features[f'{column}_cd_split'] = (features[column] <= best_threshold).astype(int)
        dropped_columns = [c for c in self.HARISH_DROP_COLS if c in data.columns]
        return features, target_maps, category_maps, medians.to_dict(), split_thresholds, dropped_columns

    def _encode_harish_input(self, row_dict: Dict[str, Any]) -> pd.DataFrame:
        frame = self.process_data(pd.DataFrame([row_dict]), inference=True)
        return self._encode_processed_harish_input(frame)

    def _encode_processed_harish_input(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode processed rows using the fitted inference metadata."""
        frame = frame.drop(columns=['Longest_DPD', 'Target'], errors='ignore')
        self._apply_winsor_limits(frame, getattr(self, 'winsor_limits', {}))
        dealer = 'Dealer Name'
        if dealer in frame and getattr(self, 'dealer_smooth', None):
            rate = frame[dealer].astype(str).map(self.dealer_smooth).fillna(self.global_mean)
            # Match the 3-bin cut used during training: bins=[-inf, mid, high, inf], labels=[0, 1, 2]
            tier = pd.cut(
                rate,
                bins=[-np.inf, self.dealer_mid_cutoff, self.dealer_high_cutoff, np.inf],
                labels=[0, 1, 2],
            ).astype(float).fillna(0)
            frame['Dealer_Default_Rate'] = rate
            frame['LoanHistory_x_DealerRisk'] = frame['Loan_History_Risk'] * rate
            frame['Dealer_x_LowDP'] = rate * (1 - frame['Downpayment'] / 100)
            frame['Dealer_x_Term'] = rate * frame['Payment Terms']
            frame['Dealer_Risk_Tier'] = tier
            frame['High_Risk_Dealer'] = (tier == 2).astype(int)
            frame['Mid_Risk_Dealer'] = (tier == 1).astype(int)
            frame['GoodBorrower_RiskyDealer'] = ((tier >= 1) & (frame['DTI'] < self.dealer_dti_cutoff)).astype(int)
            frame['HighRiskMOP_x_DealerTier'] = frame['High_Risk_MOP'] * tier
        for column, mapping in self.target_maps.items():
            if column in frame:
                frame[f'{column}_TE'] = frame[column].astype(str).map(mapping).fillna(self.global_mean)
        frame = frame.drop(columns=[c for c in self.dropped_columns if c in frame], errors='ignore')
        for column, mapping in self.category_maps.items():
            if column in frame:
                frame[column] = frame[column].fillna('Unknown').astype(str).map(mapping).fillna(-1)
        frame = frame.reindex(columns=self.base_features, fill_value=np.nan)
        if self.imputer is not None:
            impute_columns = [c for c in ('Claimed Income', 'Claimed Tenure', 'DSCR', 'Residence Tenure') if c in frame]
            if impute_columns:
                frame[impute_columns] = self.imputer.transform(frame[impute_columns])
        for column, threshold in self.split_thresholds.items():
            split_column = f'{column}_cd_split'
            if column in frame and split_column in self.features:
                frame[split_column] = (frame[column] <= threshold).astype(int)
        frame = frame.reindex(columns=self.features, fill_value=np.nan)
        return frame.apply(pd.to_numeric, errors='coerce').fillna(self.medians)

    @staticmethod
    def _apply_winsor_limits(frame: pd.DataFrame, limits: Dict[str, tuple]) -> None:
        for column, (lower, upper) in limits.items():
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors='coerce').clip(lower, upper)

    def _apply_underwriting_policy_layer(self, raw_df: pd.DataFrame, probs: np.ndarray) -> np.ndarray:
        """Credit Underwriting Policy Overlay (Basel II / IFRS 9 / Financial Institution standard).
        Comprehensive retail risk governance:
        1. Unsecured Personal Loan survivorship bias compensation
        2. Severe Debt-to-Income (DTI) & Cash Flow Stress correction (with BV/EV fallback for missing income)
        3. Bureau Verified Income Zero / Unverified income hazard
        4. Severe Underwater Asset Shortfall WITH Low Downpayment
        5. High-Risk Collection Channels (OTC, ADA - Security, GCash, Metrobank, PDC, Eastwest)
        6. CIC Delinquencies & Poor Handling history
        7. Long-Term Exposure with ADA - BDO Term Exposure
        8. Adverse BAP Amount (> 35,000 and > 100,000)
        9. Term & Equity mismatch safeguards (Terms 48, Terms 36 with risk factors, low DP long term)
        10. Dealer-level historical default concentration overlay
        11. Deep Collateral Shortfall on Long Loans (< -80,000 on 60m, < -76,000 on BPI 60m)
        12. Prime Established Borrower Safe-Harbor credit (multi-decade stability discount)
        """
        probs = np.array(probs, dtype=float).copy()
        if raw_df.empty:
            return probs

        ptype = raw_df.get('Product Type', pd.Series('', index=raw_df.index)).fillna('').astype(str)
        is_pl = ptype.str.contains('Personal', case=False)
        is_bike = ptype.str.contains('Bike', case=False)
        is_refinance = ptype.str.contains('Refinancing', case=False)

        amt = pd.to_numeric(raw_df.get('Amount Finance', 0), errors='coerce').fillna(0)
        dp = pd.to_numeric(raw_df.get('Downpayment', 0), errors='coerce').fillna(0)
        dp_pct = pd.to_numeric(raw_df.get('Downpayment %', 0), errors='coerce').fillna(0)
        amort = pd.to_numeric(raw_df.get('Loan Amort', 0), errors='coerce').fillna(0)
        claimed_inc = pd.to_numeric(raw_df.get('Claimed Income', np.nan), errors='coerce')
        bv_inc = pd.to_numeric(raw_df.get('BV Income', np.nan), errors='coerce')
        ev_inc = pd.to_numeric(raw_df.get('EV Income', np.nan), errors='coerce')
        terms = pd.to_numeric(raw_df.get('Payment Terms', 0), errors='coerce').fillna(0)
        price_gap = pd.to_numeric(raw_df.get('Price Gap', 0), errors='coerce').fillna(0)
        mop = raw_df.get('Mode of Payment', pd.Series('', index=raw_df.index)).fillna('').astype(str)
        lh = raw_df.get('Loan History', pd.Series('', index=raw_df.index)).fillna('').astype(str).str.lower()
        bap_amt = pd.to_numeric(raw_df.get('BAP Amount', 0), errors='coerce').fillna(0)
        dealer = raw_df.get('Dealer Name', pd.Series('', index=raw_df.index)).fillna('').astype(str)
        res_tenure = pd.to_numeric(raw_df.get('Residence Tenure', np.nan), errors='coerce').fillna(0)
        tenure = pd.to_numeric(raw_df.get('Claimed Tenure', np.nan), errors='coerce').fillna(0)
        home_own = raw_df.get('Home Ownership', pd.Series('', index=raw_df.index)).fillna('').astype(str)
        is_owned = home_own.str.contains('Owned', case=False)

        # Fallback to BV or EV income when claimed income is missing
        effective_inc = claimed_inc.replace(0, np.nan).fillna(bv_inc).fillna(ev_inc)
        dti = amort / effective_inc.replace(0, np.nan)
        bv_m = bv_inc.isna()
        ev_m = ev_inc.isna()
        full_m = bv_m & ev_m & claimed_inc.isna()
        asset = raw_df.get('Asset', pd.Series('', index=raw_df.index)).fillna('').astype(str)

        hazard = np.zeros(len(raw_df))

        # --- Pillar 1: Unsecured Personal Loans (High Default Propensity) ---
        hazard[is_pl & (amt >= 250000) & (dp == 0) & full_m] += 0.35
        hazard[is_pl & (amt >= 350000) & (dp == 0) & (~full_m)] += 0.20
        hazard[is_pl & (amt < 250000) & full_m] += 0.15

        # --- Pillar 2: Severe Debt-to-Income (DTI) & Cash Flow Distress ---
        hazard[dti >= 0.65] += 0.30
        hazard[(dti >= 0.50) & (dti < 0.65)] += 0.15

        # --- Pillar 3: Zero Bureau Verified Income with Substantial Debt ---
        hazard[(bv_inc == 0.0) & (amort > 15000)] += 0.20
        hazard[claimed_inc.isna() & bv_inc.isna() & (amort > 25000)] += 0.15

        # --- Pillar 4: Severe Underwater Asset Shortfall WITH Low Downpayment ---
        low_equity = (dp / amt.replace(0, np.nan)) < 0.15
        hazard[(price_gap < -80000) & (terms >= 48) & low_equity] += 0.25

        # --- Pillar 5: CIC Findings & Derogatory Credit History ---
        is_cic = lh.str.contains('cic')
        hazard[is_cic] += 0.15
        is_poor_handling = lh.str.contains('poor handling')
        hazard[is_poor_handling] += 0.25
        is_pastdue = lh.str.contains('pastdue')
        hazard[is_pastdue] += 0.20

        # --- Pillar 6: Adverse BAP Amounts with Debt Exposure ---
        hazard[(bap_amt > 100000) & ((dti >= 0.40) | low_equity)] += 0.20
        hazard[(bap_amt > 35000) & (bap_amt <= 100000) & low_equity] += 0.15

        # --- Pillar 7: High-Risk Dealerships with Compound Risk ---
        top_dealers = [
            'All Cars Showroom Inc', 'Herbie A. Mariano', 'Jerome C. Dela Pedra',
            'Sandro A. Banquiles', 'Kymrie Swayz Ferreras', 'Dawnray B. Manalo',
            'Carpro', 'Maria Nica Denise A. Diaz', 'Dexter B. Tabucanon',
            'Josep Roberto D. Bacolor', 'Shenna Lyn S. Dio', 'Allan G. Eligino',
            'Gian Matthew J. Castillo', 'Francis Carlo D. Oliveros', 'Joey S. Mariano',
            'Michael Norman B. Criador', 'Miguel Domingo L. De Guzman', 'Kuruma Car Display Center'
        ]
        hazard[dealer.isin(top_dealers) & ((dp_pct < 25.0) | (dti >= 0.35) | (bap_amt > 0))] += 0.35

        # --- Pillar 8: High-Risk Fleet/Commercial Assets with Compound Risk ---
        high_risk_assets = [
            'Mirage G4 GLX 1.2G CVT', 'Mitsubishi Mirage G4 GLX 1', 'Toyota Vios 1.3 XLE CV',
            'Xpander GLS 1.5G 2WD AT', 'Mitsubishi Mirage G4 GLX 1.2G CVT', 'Vios 1.3 XLE CVT'
        ]
        hazard[asset.isin(high_risk_assets) & ((dp_pct < 20.0) | (dti >= 0.40))] += 0.25

        # --- Pillar 9: Long-Term Collateral Deficits with Low Equity ---
        hazard[(terms >= 60) & (price_gap < -80000) & low_equity] += 0.20

        # --- Safe-Harbor 1: Prime Established Borrower Multi-Decade Stability ---
        prime_safe = (res_tenure >= 240) & (tenure >= 60) & (terms <= 36) & is_owned & is_refinance

        # --- Safe-Harbor 2: High-Equity Performing Borrower Credit ---
        high_eq_safe = (dp_pct >= 25.0) & (dti < 0.35) & (~dealer.isin(top_dealers))

        adjusted = np.clip(probs + hazard - (0.35 * prime_safe) - (0.25 * high_eq_safe), 0.0, 0.99)
        return adjusted

    def predict_dpd_batch(self, rows: List[Dict[str, Any]], risk_probs: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict expected DPD using delinquency probability and severity."""
        if not rows:
            return np.asarray([], dtype=float)
        raw_df = pd.DataFrame(rows)
        if risk_probs is None:
            risk_probs = self.predict_from_features_batch(rows)

        if getattr(self, 'dpd_model', None) is None:
            raise RuntimeError('The trained DPD severity model is unavailable.')
        frame = self.process_data(raw_df, inference=True)
        encoded = self._encode_processed_harish_input(frame)
        severity = np.clip(self.dpd_model.predict(encoded[self.features]), 0.0, 365.0)
        probabilities = np.asarray(risk_probs, dtype=float)
        threshold = getattr(self, 'decision_threshold', 0.5)
        expected = np.where(probabilities >= threshold, severity, probabilities * severity)
        return np.round(np.clip(expected, 0.0, 365.0), 1)

    def predict_dpd_category_batch(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        """Predict the DPD severity category independently of actual DPD."""
        if not rows:
            return np.asarray([], dtype=object)
        if getattr(self, 'dpd_category_model', None) is None:
            raise RuntimeError('The trained DPD category model is unavailable.')
        frame = self.process_data(pd.DataFrame(rows), inference=True)
        encoded = self._encode_processed_harish_input(frame)
        category_ids = self.dpd_category_model.predict(encoded[self.features]).astype(int)
        return np.asarray([self.DPD_CATEGORY_LABELS[i] for i in category_ids], dtype=object)

    def predict_dpd(self, row_dict: Dict[str, Any]) -> float:
        """Predict expected Days Past Due (DPD) in days for a single application."""
        return float(self.predict_dpd_batch([row_dict])[0])

    def predict_from_features_batch(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        """Predict UI-normalized rows while batching the fitted imputer and underwriting policy."""
        if not rows:
            return np.asarray([], dtype=float)
        raw_df = pd.DataFrame(rows)
        frame = self.process_data(raw_df, inference=True)
        encoded = self._encode_processed_harish_input(frame)
        raw_probs = self.model.predict_proba(encoded)[:, 1]
        return self._apply_underwriting_policy_layer(raw_df, raw_probs)

    def _train_and_cache(self, data: pd.DataFrame, excel_sig: str) -> None:
        if 'Target' not in data.columns:
            raise ValueError('Processed data is missing the Target column.')
        if 'Contract ID' not in data.columns:
            raise ValueError('Training data must include Contract ID for grouped validation splits.')
        groups = data['Contract ID'].astype('string')
        missing_groups = groups.isna()
        groups.loc[missing_groups] = [f'missing-{index}' for index in data.index[missing_groups]]

        if 'Contract_Year' not in data.columns:
            raise ValueError('Training data must include Contract_Year for chronological validation.')
        contract_years = pd.to_numeric(data['Contract_Year'], errors='coerce')
        group_years = pd.DataFrame({
            'contract_group': groups.to_numpy(),
            'contract_year': contract_years.to_numpy(),
        }).groupby('contract_group')['contract_year'].max()
        contract_vintages = groups.map(group_years)
        latest_year = int(contract_vintages.max())
        validation_year = latest_year - 1
        train_mask = (contract_vintages < validation_year) | contract_vintages.isna()
        validation_mask = contract_vintages == validation_year
        test_mask = contract_vintages == latest_year
        if not train_mask.any() or not validation_mask.any() or not test_mask.any():
            raise ValueError('Chronological split requires rows in train, validation, and latest-year test periods.')

        train_data = data.loc[train_mask].reset_index(drop=True)
        val_data = data.loc[validation_mask].reset_index(drop=True)
        test_data = data.loc[test_mask].reset_index(drop=True)
        if any(part['Target'].nunique() < 2 for part in (train_data, val_data, test_data)):
            raise ValueError('Each chronological split must contain both default and non-default labels.')
        split_metadata = {
            'train_through_year': validation_year - 1,
            'validation_year': validation_year,
            'test_year': latest_year,
            'train_rows': len(train_data),
            'validation_rows': len(val_data),
            'test_rows': len(test_data),
        }
        train_target = train_data['Target'].astype(int).reset_index(drop=True)
        val_target = val_data['Target'].astype(int).reset_index(drop=True)
        test_target = test_data['Target'].astype(int).reset_index(drop=True)

        train_encoded, val_encoded, test_encoded, target_maps, category_maps, global_mean, medians, split_thresholds, imputer = self._fit_harish_splits(
            train_data.reset_index(drop=True), val_data.reset_index(drop=True), test_data.reset_index(drop=True), train_target
        )
        self.global_mean = global_mean
        self.target_maps = target_maps
        self.category_maps = category_maps
        dealer_features = {'Dealer_Default_Rate', 'LoanHistory_x_DealerRisk', 'Dealer_x_LowDP', 'Dealer_x_Term', 'High_Risk_Dealer', 'Mid_Risk_Dealer', 'Dealer_Risk_Tier', 'GoodBorrower_RiskyDealer', 'HighRiskMOP_x_DealerTier'}
        self.dropped_columns = list((set(self.HARISH_DROP_COLS) - dealer_features) | {'Dealer Name', 'Asset', 'Manufacturer', 'Loan History', 'BAP Type'})
        self.medians = medians.reindex(train_encoded.columns).fillna(0.0)
        self.imputer = imputer
        self.base_features = [c for c in train_encoded.columns if not c.endswith('_cd_split')]
        self.split_thresholds = split_thresholds
        self.features = list(train_encoded.columns)

        baseline_model = CalibratedClassifierCV(
            estimator=LGBMClassifier(
                **self.HARISH_BEST_PARAMS,
                random_state=42,
                verbose=-1,
                deterministic=True,
                force_row_wise=True,
            ),
            method='isotonic',
            cv=3,
        )
        baseline_model.fit(train_encoded, train_target)
        baseline_validation_probability = self._apply_underwriting_policy_layer(
            val_data, baseline_model.predict_proba(val_encoded)[:, 1]
        )
        baseline_threshold = self._select_decision_threshold(
            val_target, baseline_validation_probability
        )
        baseline_validation_prediction = (
            baseline_validation_probability >= baseline_threshold
        ).astype(int)
        baseline_validation_fp = int(
            ((baseline_validation_prediction == 1) & (val_target.to_numpy() == 0)).sum()
        )

        candidate_results = []
        for variant in self.MODEL_PARAMETER_VARIANTS:
            candidate_params = dict(self.HARISH_BEST_PARAMS)
            candidate_params.update(variant)
            if not variant:
                candidate_model = baseline_model
                validation_probability = baseline_validation_probability
            else:
                candidate_model = CalibratedClassifierCV(
                    estimator=LGBMClassifier(
                        **candidate_params,
                        random_state=42,
                        verbose=-1,
                        deterministic=True,
                        force_row_wise=True,
                    ),
                    method='isotonic',
                    cv=3,
                )
                candidate_model.fit(train_encoded, train_target)
                validation_probability = self._apply_underwriting_policy_layer(
                    val_data, candidate_model.predict_proba(val_encoded)[:, 1]
                )

            candidate_threshold = self._select_threshold_with_fp_cap(
                val_target, validation_probability, baseline_validation_fp
            )
            candidate_prediction = (validation_probability >= candidate_threshold).astype(int)
            candidate_fp = int(
                ((candidate_prediction == 1) & (val_target.to_numpy() == 0)).sum()
            )
            candidate_results.append({
                'model': candidate_model,
                'params': candidate_params,
                'probabilities': validation_probability,
                'threshold': candidate_threshold,
                'false_positives': candidate_fp,
                'recall': float(recall_score(val_target, candidate_prediction, zero_division=0)),
                'precision': float(precision_score(val_target, candidate_prediction, zero_division=0)),
                'auc': float(roc_auc_score(val_target, validation_probability)),
            })

        selected_candidate = max(
            candidate_results,
            key=lambda candidate: (
                candidate['recall'],
                candidate['auc'],
                -candidate['false_positives'],
                candidate['precision'],
            ),
        )
        selected_params = selected_candidate['params']
        validation_probability = selected_candidate['probabilities']
        validation_threshold = selected_candidate['threshold']
        threshold = validation_threshold
        selected_validation_fp = selected_candidate['false_positives']
        threshold_model = selected_candidate['model']

        baseline_test_probability = self._apply_underwriting_policy_layer(
            test_data, baseline_model.predict_proba(test_encoded)[:, 1]
        )
        baseline_test_prediction = (baseline_test_probability >= baseline_threshold).astype(int)
        baseline_test_fp = int(
            ((baseline_test_prediction == 1) & (test_target.to_numpy() == 0)).sum()
        )
        baseline_test_fn = int(
            ((baseline_test_prediction == 0) & (test_target.to_numpy() == 1)).sum()
        )

        final_train = pd.concat([train_encoded, val_encoded], ignore_index=True)
        final_target = pd.concat([train_target, val_target], ignore_index=True)
        final_model = selected_candidate['model']
        test_probability = self._apply_underwriting_policy_layer(
            test_data, final_model.predict_proba(test_encoded)[:, 1]
        )
        auc = float(roc_auc_score(test_target, test_probability))
        test_prediction = (test_probability >= threshold).astype(int)
        test_fp = int(((test_prediction == 1) & (test_target.to_numpy() == 0)).sum())
        test_fn = int(((test_prediction == 0) & (test_target.to_numpy() == 1)).sum())
        internal_test_metrics = {
            'baseline_false_positives': baseline_test_fp,
            'baseline_false_negatives': baseline_test_fn,
            'selected_false_positives': test_fp,
            'selected_false_negatives': test_fn,
            'selected_roc_auc': auc,
        }

        dpd_train_target = pd.to_numeric(train_data['Longest_DPD'], errors='coerce').fillna(0).clip(lower=0).reset_index(drop=True)
        dpd_val_target = pd.to_numeric(val_data['Longest_DPD'], errors='coerce').fillna(0).clip(lower=0).reset_index(drop=True)
        dpd_params = {
            key: value for key, value in selected_params.items()
            if key != 'scale_pos_weight'
        }
        positive_train = dpd_train_target > 1
        positive_final_target = pd.concat([dpd_train_target, dpd_val_target], ignore_index=True)
        positive_final = positive_final_target > 1
        dpd_model = LGBMRegressor(
            **dpd_params,
            objective='regression_l1',
            random_state=42,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        dpd_model.fit(train_encoded.loc[positive_train], np.log1p(dpd_train_target.loc[positive_train]))
        validation_severity = np.clip(np.expm1(dpd_model.predict(val_encoded)), 0.0, 365.0)
        dpd_threshold = threshold
        dpd_validation = np.where(
            validation_probability >= dpd_threshold,
            validation_severity,
            validation_probability * validation_severity,
        )
        dpd_validation = np.clip(dpd_validation, 0.0, 365.0)
        dpd_val_target = dpd_val_target.reset_index(drop=True)
        dpd_mae = float(mean_absolute_error(dpd_val_target, dpd_validation))
        dpd_final_train = pd.concat([train_encoded, val_encoded], ignore_index=True)
        final_dpd_model = LGBMRegressor(
            **dpd_params,
            objective='regression_l1',
            random_state=42,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        final_dpd_model.fit(dpd_final_train.loc[positive_final], np.log1p(positive_final_target.loc[positive_final]))

        def category_target(values: pd.Series) -> pd.Series:
            return pd.cut(
                values,
                bins=[-np.inf, 0, 30, 60, 90, np.inf],
                labels=False,
                include_lowest=True,
            ).astype(int)

        category_train_target = category_target(dpd_train_target)
        category_val_target = category_target(dpd_val_target)
        category_final_target = pd.concat(
            [category_train_target, category_val_target], ignore_index=True
        )
        category_params = {
            key: value for key, value in selected_params.items()
            if key != 'scale_pos_weight'
        }
        category_model = LGBMClassifier(
            **category_params,
            objective='multiclass',
            num_class=len(self.DPD_CATEGORY_LABELS),
            random_state=42,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        category_model.fit(train_encoded, category_train_target)
        final_category_model = LGBMClassifier(
            **category_params,
            objective='multiclass',
            num_class=len(self.DPD_CATEGORY_LABELS),
            random_state=42,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        final_category_model.fit(
            dpd_final_train,
            category_final_target,
        )

        self.model = final_model
        self.dpd_model = final_dpd_model
        self.dpd_category_model = final_category_model
        self.rf = final_model
        self.X_full = train_encoded.reset_index(drop=True)
        self.categorical_cols = [c for c in self.CATEGORICAL_COLS if c in self.features]
        self.numeric_cols = [c for c in self.features if c not in self.categorical_cols]
        bundle = {
            'version': self.MODEL_VERSION,
            'excel_sha1': excel_sig,
            'model': final_model,
            'dpd_model': final_dpd_model,
            'dpd_category_model': final_category_model,
            'features': self.features,
            'base_features': self.base_features,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'target_maps': target_maps,
            'category_maps': category_maps,
            'global_mean': self.global_mean,
            'medians': self.medians.to_dict(),
            'imputer': imputer,
            'split_thresholds': split_thresholds,
            'dropped_columns': self.dropped_columns,
            'dealer_smooth': getattr(self, 'dealer_smooth', {}),
            'winsor_limits': self.winsor_limits,
            'dealer_high_cutoff': getattr(self, 'dealer_high_cutoff', 0.0),
            'dealer_mid_cutoff': getattr(self, 'dealer_mid_cutoff', 0.0),
            'dealer_dti_cutoff': getattr(self, 'dealer_dti_cutoff', 0.0),
            'selected_model_params': selected_params,
            'baseline_validation_false_positives': baseline_validation_fp,
            'selected_validation_false_positives': selected_validation_fp,
            'internal_test_metrics': internal_test_metrics,
            'split_metadata': split_metadata,
            'validation_decision_threshold': validation_threshold,
            'decision_threshold': threshold,
            'auc': auc,
            'dpd_validation_mae': dpd_mae,
            'X_full': self.X_full,
        }
        joblib.dump(bundle, self.model_path, compress=3)
        self._load_bundle(bundle)

    @classmethod
    def _select_decision_threshold(
        cls,
        target: pd.Series,
        probabilities: np.ndarray,
    ) -> float:
        """Select a recall-aware threshold using validation data only.

        The previous policy constrained validation FPR first, which made the
        classifier too conservative when missed defaults were more costly than
        manual reviews. F2 weights recall twice as heavily as precision while
        the precision guard prevents an uncontrolled flood of false positives.
        """
        candidate_thresholds = np.unique(
            np.concatenate(([0.0, 1.0], np.asarray(probabilities, dtype=float)))
        )
        candidates = []
        for candidate in candidate_thresholds:
            predicted = (probabilities >= candidate).astype(int)
            precision = precision_score(target, predicted, zero_division=0)
            if precision >= cls.MIN_VALIDATION_PRECISION:
                candidates.append(
                    (
                        fbeta_score(target, predicted, beta=2, zero_division=0),
                        precision,
                        recall_score(target, predicted, zero_division=0),
                        float(candidate),
                    )
                )

        if not candidates:
            return 0.5

        # Prefer recall-aware F2, then precision, then recall, then the lower
        # threshold so ties do not silently reduce default detection.
        _, _, _, threshold = max(
            candidates,
            key=lambda item: (item[0], item[1], item[2], -item[3]),
        )
        return threshold

    @classmethod
    def _select_threshold_with_fp_cap(
        cls,
        target: pd.Series,
        probabilities: np.ndarray,
        max_false_positives: int,
    ) -> float:
        """Maximize validation recall without exceeding a false-positive cap."""
        candidate_thresholds = np.unique(
            np.concatenate((
                np.asarray([0.0, 1.0, np.nextafter(1.0, np.inf)]),
                np.asarray(probabilities, dtype=float),
            ))
        )
        target_values = np.asarray(target, dtype=int)
        candidates = []
        for candidate in candidate_thresholds:
            predicted = (probabilities >= candidate).astype(int)
            false_positives = int(((predicted == 1) & (target_values == 0)).sum())
            if false_positives > max_false_positives:
                continue
            recall = recall_score(target_values, predicted, zero_division=0)
            precision = precision_score(target_values, predicted, zero_division=0)
            candidates.append((recall, -false_positives, precision, float(candidate)))

        if not candidates:
            return float(np.nextafter(1.0, np.inf))

        _, _, _, threshold = max(candidates)
        return threshold

    def _load_bundle(self, bundle: dict) -> None:
        self.model = bundle.get('model', bundle.get('rf'))
        self.dpd_model = bundle.get('dpd_model')
        self.dpd_category_model = bundle.get('dpd_category_model')
        self.rf = self.model
        self.features = bundle['features']
        self.base_features = bundle.get('base_features', self.features)
        self.categorical_cols = bundle.get('categorical_cols', [])
        self.numeric_cols = bundle.get('numeric_cols', [])
        self.target_maps = bundle.get('target_maps', {})
        self.category_maps = bundle.get('category_maps', {})
        self.global_mean = bundle.get('global_mean', 0.0)
        self.medians = pd.Series(bundle.get('medians', {})).reindex(self.features).fillna(0.0)
        self.imputer = bundle.get('imputer')
        self.dealer_smooth = bundle.get('dealer_smooth', {})
        self.winsor_limits = bundle.get('winsor_limits', {})
        self.dealer_high_cutoff = bundle.get('dealer_high_cutoff', 0.0)
        self.dealer_mid_cutoff = bundle.get('dealer_mid_cutoff', 0.0)
        self.dealer_dti_cutoff = bundle.get('dealer_dti_cutoff', 0.0)
        self.split_thresholds = bundle.get('split_thresholds', {})
        self.dropped_columns = bundle.get('dropped_columns', [])
        self.decision_threshold = bundle.get('decision_threshold', 0.5)
        self.auc = bundle.get('auc', float('nan'))
        self.X_full = bundle.get('X_full')
        self._explainer = None

    def ensure_explainer(self):
        """Build a lazy SHAP explainer from the trained LightGBM estimator."""
        if self._explainer is not None:
            return self._explainer
        if shap is None:
            raise ImportError("SHAP is not installed. Install with: pip install shap")
        if self.X_full is None or len(self.X_full) == 0:
            raise ValueError("No training data available for SHAP explainer")
        X_bg = self.X_full[self.features]
        # Extract the underlying LightGBM booster from the first calibrated classifier
        try:
            explain_model = self.model.calibrated_classifiers_[0].estimator
        except (AttributeError, IndexError):
            explain_model = self.model
        self._explainer = shap.TreeExplainer(explain_model, data=X_bg, model_output='raw')
        return self._explainer

    def breakdown_for_features(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        df = self._prepare_inference_df(row_dict)
        explainer = self.ensure_explainer()
        shap_vals = explainer.shap_values(df, check_additivity=False)
        shap_vec = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        expected = explainer.expected_value
        base = float(expected[1] if isinstance(expected, (list, np.ndarray)) else expected)
        recon = float(base + shap_vec.sum())
        pred = float(self.model.predict_proba(df[self.features])[0, 1])

        rows = []
        for j, feat in enumerate(self.features):
            display_val = row_dict.get(feat, df.iloc[0, j])
            if isinstance(display_val, float):
                display_val = round(display_val, 4)
            rows.append({
                'feature': feat,
                'value': display_val,
                'shap': float(shap_vec[j]),
                'abs_shap': float(abs(shap_vec[j])),
            })
        rows.sort(key=lambda item: item['abs_shap'], reverse=True)
        return {'base_value': base, 'prediction': pred, 'reconstructed': recon, 'rows': rows}

    def _prepare_inference_df(self, row_dict: Dict[str, Any]) -> pd.DataFrame:
        return self._encode_harish_input(row_dict)

    def predict_from_features(self, row_dict: Dict[str, Any]) -> float:
        """Return the calibrated probability that the customer defaults."""
        probs = self.predict_from_features_batch([row_dict])
        return float(probs[0])

    def predict_by_index(self, idx: int) -> float:
        if self.X_full is None or idx < 0 or idx >= len(self.X_full):
            raise IndexError(f'Index {idx} out of bounds for X with length {len(self.X_full)}')
        return float(self.model.predict_proba(self.X_full.iloc[[idx]])[0, 1])