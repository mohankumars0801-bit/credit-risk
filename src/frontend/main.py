import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tkinter as tk
import tkinter.font as tkFont
from tkinter import font, messagebox, filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from src.backend.ai_model import AIModel, resource_path
from src.backend.db import fetch_ci_mock_crifv4, load_config, save_config, get_connection, get_connection_local
from datetime import datetime, timedelta, date
import threading
from src.frontend.splash import Splash
from src.backend.users import ensure_user_store, authenticate
from src.frontend.admin_dashboard import AdminDashboard
from src.backend.logger import get_logger, log_prediction
import pandas as pd

logger = get_logger(__name__)


def read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    """Read a CSV or Excel file trying appropriate parsers, encodings, and separators.

    Handles:
    1. Excel files (XLSX / XLS / OLE2 / ZIP) even if misnamed with a .csv extension.
    2. Encoding errors (UnicodeDecodeError) - Excel on Windows saves CSV in cp1252 or utf-8.
    3. Tokenization errors (ParserError) - mismatched number of fields per row,
       wrong delimiter (semicolon, tab vs comma), or unquoted commas in values.
    """
    # --- Step 1: Detect if file is an Excel binary workbook (XLSX zip container or legacy XLS) ---
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
        # b"PK\x03\x04" is ZIP / XLSX; b"\xd0\xcf\x11\xe0" is legacy XLS (OLE2 Compound Document)
        if magic.startswith(b"PK\x03\x04") or magic.startswith(b"\xd0\xcf\x11\xe0"):
            return pd.read_excel(path)
    except Exception:
        pass

    # --- Step 2: Check standard Excel extension ---
    if path.lower().endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return pd.read_excel(path)
        except Exception:
            pass

    # --- Step 3: Text CSV parsing strategies ---
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_err: Exception = RuntimeError("No encodings tried")

    # Pass 1: standard comma-separated, each encoding
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:          # ParserError or other tokenization issues
            last_err = exc

    # Pass 2: auto-detect separator (handles ; and \t delimited files)
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, sep=None, engine="python", **kwargs)
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc

    # Pass 3: skip bad lines as last resort
    for enc in encodings:
        try:
            return pd.read_csv(
                path, encoding=enc, sep=None, engine="python",
                on_bad_lines="skip", **kwargs
            )
        except UnicodeDecodeError as exc:
            last_err = exc
        except Exception as exc:
            last_err = exc

    # Pass 4: Final fallback attempt to read via Excel in case file was an Excel file
    try:
        return pd.read_excel(path)
    except Exception:
        pass

    raise last_err



class PlaceholderEntry(tk.Entry):
    """Entry widget that shows a greyed-out placeholder when empty."""

    def __init__(self, master=None, placeholder="", color="grey", **kwargs):
        # Extract 'show' before passing to super so we can toggle it
        self._real_show = kwargs.pop("show", "")
        super().__init__(master, **kwargs)
        self.placeholder = placeholder
        self.placeholder_color = color
        self.default_fg = self["fg"]
        self.bind("<FocusIn>", self._clear)
        self.bind("<FocusOut>", self._show)
        self._show()

    def _show(self, *_):
        if not self.get() or self["fg"] == self.placeholder_color:
            # Show placeholder: disable masking so text is readable
            self.configure(show="", fg=self.placeholder_color)
            self.delete(0, tk.END)
            self.insert(0, self.placeholder)

    def _clear(self, *_):
        if self["fg"] == self.placeholder_color:
            self.configure(show=self._real_show, fg=self.default_fg)
            self.delete(0, tk.END)

class App(tk.Tk):
    GREEN, BG_GRAY = "#2E7D32", "#F0F0F0"
    ENTRY_BG, ENTRY_FG = "#FFFFFF", "#000000"
    BORDER = "#DDD"

    def __init__(self):
        super().__init__()
        ensure_user_store()
        self.withdraw()  # hide main window until we’re ready

        # Show splash immediately (white, JACCS primary, small “Powered by LTID”)
        self.splash = Splash(
            self,
            jaccs_logo=resource_path("src/data/img/jaccs_logo.png"),
            ltid_logo=resource_path("src/data/img/ltid_logo.png"),
            title="Credit Scoring System",
            message="Initializing model…"
        )

        # Light-weight window config can happen now
        self.title("JACCS Credit Scoring System")
        self.configure(bg=self.BG_GRAY)
        self.geometry("800x850")
        self.resizable(True, True)

        # Defer heavy work to a background thread
        self.ai = None
        self.logo = None
        self._cached_local_df = None
        threading.Thread(target=self._heavy_init, daemon=True).start()

    def _heavy_init(self):
        """Heavy work: train/load the model, load images, etc. (NO Tk calls here)."""
        try:
            # 1) Load & train AI model (heavy)
            # Pass relative path so AIModel can resolve it correctly for both dev and frozen
            self.ai = AIModel("src/data/May 12 Status Contracts.xlsx","src/data/Extracted New.xlsx")

            # 2) Load header logo image (pure PIL work is OK here)
            img = Image.open(resource_path("src/data/img/logo.png")).resize((180, 70), Image.LANCZOS)

            # Hand off any Tk object creation + UI build to the main thread
            self.after(0, lambda pil_img=img: self._finish_init(pil_img))
        except Exception as e:
            # Close splash and show an error if something goes wrong
            error_msg = str(e)
            self.after(0,
                       lambda: (self.splash.destroy(), tk.messagebox.showerror("Init Error", error_msg), self.destroy()))
            
    def _finish_init(self, pil_img):
        """Back on the main thread: build Tk widgets, close splash, show main window."""
        # Convert PIL image to Tk PhotoImage on the main thread
        self.logo = ImageTk.PhotoImage(pil_img)

        # Initialize state
        self.db_row = None
        self.ai_features = None
        self.contract_date_defaulted = False
        self._edit_entry = None  # inline editor widget
        self._editing = {"item": None, "column": None}

        # Build UI
        self._build_scoring()
        self.scoring_frame.place_forget()
        self._build_login()

        # Close splash & show the main window
        if self.splash and self.splash.winfo_exists():
            self.splash.destroy()
        self.deiconify()

    def _decorate(self, w):
        w.configure(
            bd=0, relief="flat",
            highlightthickness=1,
            highlightbackground=self.BORDER,
            highlightcolor=self.BORDER
        )
    
    # -- Login -- #
    def _build_login(self):
        f = self.login_frame = tk.Frame(self, bg="white")
        f.place(relx=0.5, rely=0.5, anchor="center", width=420, height=440)

        tk.Label(f, image=self.logo, bg="white").pack(pady=(30, 5))
        tk.Label(f, text="Please Sign In", bg="white", fg=self.GREEN,
                 font=font.Font(size=16, weight="bold")).pack(pady=(0, 20))

        self.login_email = PlaceholderEntry(
            f, placeholder="Email", fg=self.ENTRY_FG, bg=self.ENTRY_BG,
            insertbackground=self.ENTRY_FG, font=font.Font(size=12)
        )
        self.login_email.pack(fill="x", padx=40, ipady=8, pady=5)
        self._decorate(self.login_email)

        self.login_pass = PlaceholderEntry(
            f, placeholder="Password", fg=self.ENTRY_FG, bg=self.ENTRY_BG,
            show="*", insertbackground=self.ENTRY_FG, font=font.Font(size=12)
        )
        self.login_pass.pack(fill="x", padx=40, ipady=8, pady=5)
        self._decorate(self.login_pass)

        btn = tk.Button(
            f, text="Sign In", command=self._on_signin,
            bg=self.GREEN, fg="black",
            activebackground=self.GREEN, activeforeground="black",
            font=font.Font(size=12), cursor="hand2"
        )
        btn.pack(fill="x", padx=40, ipady=8, pady=(20, 5))
        self._decorate(btn)

    def _on_signin(self):
        email = self.login_email.get().strip()
        pwd = self.login_pass.get().strip()
        role = authenticate(email, pwd)
        if not role:
            return messagebox.showerror("Login Failed", "Invalid credentials")

        self.login_frame.place_forget()
        if role.lower() == "admin":
            self._show_admin()
        else:
            self._show_scoring()

    def _show_scoring(self):
        # hide admin and multiple frames if they exist
        if hasattr(self, "admin_frame"):
            self.admin_frame.place_forget()
        if hasattr(self, "multiple_frame"):
            self.multiple_frame.place_forget()
        if hasattr(self, "scoring_frame"):
            self.scoring_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Restore original scoring window size
        self.geometry("800x850")

    def _show_admin(self):
        if not hasattr(self, "admin_frame"):
            self.admin_frame = AdminDashboard(self, on_open_scoring=self._show_scoring, bg=self.BG_GRAY)
        self.admin_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def _select_batch_file(self):
        # Reset UI to initial state
        self._batch_table_frame.pack_forget()
        for child in self._batch_table_frame.winfo_children():
            child.destroy()
        
        # Show upload frame and description again
        self.upload_frame.pack(pady=(0, 10), padx=16, fill="x")
        self.desc_label.pack(anchor="center", pady=(0, 10))
        
        # Re-enable browse and upload buttons
        self.btn_browse.config(state="normal")
        self.btn_upload.config(text="Upload", command=self._load_batch_file, bg=self.GREEN, state="normal")
        self._batch_path_entry.config(state="normal")
        
        self._batch_df = None
        self._batch_results = []
        self._batch_file_path.set("")

        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("All supported (*.csv;*.xlsx;*.xls)", "*.csv;*.xlsx;*.xls"),
                ("CSV file (*.csv)", "*.csv"),
                ("Excel file (*.xlsx;*.xls)", "*.xlsx;*.xls"),
                ("All files (*.*)", "*.*"),
            ],
        )
        if file_path:
            self._batch_file_path.set(file_path)
            if hasattr(self, "_batch_path_entry"):
                self._batch_path_entry.delete(0, tk.END)
                self._batch_path_entry.insert(0, file_path)
            # Change button back to Upload when Browse is clicked
            self.btn_upload.config(text="Upload", command=self._load_batch_file, bg=self.GREEN)

    def _load_batch_file(self):
        file_path = self._batch_file_path.get().strip()
        if not file_path and hasattr(self, "_batch_path_entry"):
            file_path = self._batch_path_entry.get().strip()

        if not file_path:
            return messagebox.showwarning("Missing Path", "Please select or type the file path.")

        if not os.path.exists(file_path):
            return messagebox.showerror("File Not Found", f"Cannot find file:\n{file_path}")

        try:
            df = read_csv_safe(file_path)
            print('read file successfully')
        except Exception as e:
            return messagebox.showerror("Load Error", f"Could not load file:\n{e}")

        required = {"Application No", "Party No"}
        if not self.check_file(df, required):
            return messagebox.showerror("Invalid File",
                                        f"Expected columns: {required}\nGot: {list(df.columns)}\n\nMake sure row 1 is the header.")

        self._batch_df = df
        self._batch_results = []

        messagebox.showinfo("Loaded", f"File loaded with {len(df)} rows.")

        # Hide description after successful load
        self.desc_label.pack_forget()
        
        # Keep upload frame visible with updated button states
        self.btn_upload.config(text="Prediction", command=self._batch_featch_data, bg='orange', state="normal")
        self._batch_path_entry.config(state="readonly")

        # Pack the batch table frame to show preview
        self._batch_table_frame.pack(fill='both', expand=True, padx=16, pady=(6, 10))
        
        # Show preview of uploaded data
        self._render_batch_table()

    def _forgot_pw(self):
        top = tk.Toplevel(self); top.title("Reset Password")
        top.geometry("350x180"); top.configure(bg="white")
        tk.Label(top, text="Enter your email:", bg="white",
                 font=font.Font(size=12)).pack(pady=(20, 5))
        ent = PlaceholderEntry(top, placeholder="Email",
                               fg=self.ENTRY_FG, bg=self.ENTRY_BG,
                               insertbackground=self.ENTRY_FG,
                               font=font.Font(size=12))
        ent.pack(fill="x", padx=30, ipady=6)
        self._decorate(ent)
        def send():
            messagebox.showinfo("Password Reset",
                                f"If {ent.get()} exists, a reset link was sent.")
            top.destroy()
        btn = tk.Button(top, text="Send Reset Link", command=send,
                        bg=self.GREEN, fg="black",
                        font=font.Font(size=12), cursor="hand2")
        btn.pack(pady=(20, 0), ipady=6, ipadx=10)
        self._decorate(btn)

    # —— CREDIT SCORING SCREEN —— #
    def _build_scoring(self):
        self.geometry("800x850")
        if hasattr(self, 'multiple_frame'):
            self.multiple_frame.place_forget()
        f = self.scoring_frame = tk.Frame(self, bg=self.BG_GRAY)
        f.place(relx=0, rely=0, relwidth=1, relheight=1)

        card = tk.Frame(f, bg="white")
        card.place(relx=0.5, rely=0.5, anchor="center", width=740, height=780)

        tk.Label(card, image=self.logo, bg="white").pack(pady=(20, 5))
        # Option buttion to compute all at once
        com_all_btn = tk.Button(card, text="≣", command= self._batch_processing, 
                                   bg='white', fg='black', bd=0,
                                   font=font.Font(size=24), cursor="hand2") # ⁘⇶≡
        com_all_btn.place(relx=0.0,rely=0.0, anchor='nw', x=8, y=0)

        # Tooltip for batch button
        self.tooltip_label = tk.Label(card, text="", bg="lightyellow", fg="black", font=font.Font(size=10), bd=1, relief="solid")
        self.tooltip_label.place_forget()
        com_all_btn.bind("<Enter>", lambda e: self._show_tooltip(e, "Switch to Batch Processing Mode"))
        com_all_btn.bind("<Leave>", self._hide_tooltip)

        # gear/settings button
        gear_btn = tk.Button(card, text="⚙", command=self._open_db_settings,
                             bg='white', fg="black", bd=0,
                             font=font.Font(size=14), cursor="hand2")
        gear_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        tk.Label(card, text="Credit Score Calculator", bg="white", fg=self.GREEN,
                 font=font.Font(size=18, weight="bold")).pack(pady=(0, 15))

        # Search (Party No + Application No)
        frm = tk.Frame(card, bg="white"); frm.pack(pady=(0, 10))
        self.ent_party = PlaceholderEntry(
            frm, placeholder="Party No (e.g., 201614)", fg=self.ENTRY_FG, bg=self.ENTRY_BG,
            insertbackground=self.ENTRY_FG, font=font.Font(size=12), width=20
        )
        self.ent_party.pack(side="left", ipady=6, padx=(0, 8))
        self._decorate(self.ent_party)

        self.ent_app = PlaceholderEntry(
            frm, placeholder="Application No (e.g., 2)", fg=self.ENTRY_FG, bg=self.ENTRY_BG,
            insertbackground=self.ENTRY_FG, font=font.Font(size=12), width=20
        )
        self.ent_app.pack(side="left", ipady=6, padx=(0, 8))
        self._decorate(self.ent_app)

        # Checkbox for CSV mode
        self.csv_mode = tk.BooleanVar()
        tk.Checkbutton(frm, text="Load from CSV", variable=self.csv_mode, bg="white").pack(side="left", padx=(10, 0))

        btn_fetch = tk.Button(frm, text="Fetch", command=self.fetch_customer,
                              bg=self.GREEN, fg="black",
                              activebackground=self.GREEN, activeforeground="black",
                              font=font.Font(size=12), cursor="hand2")
        btn_fetch.pack(side="left", ipady=3, ipadx=10)
        self._decorate(btn_fetch)

        # Details area (AI features only)
        tk.Label(card, text="AI Feature Inputs", bg="white", fg=self.GREEN,
                 font=font.Font(size=14, weight="bold")).pack(pady=(12, 2))
        tk.Label(card, text="(Double-click a Value to edit. Leave blank = Missing.)",
                 bg="white", fg="#555", font=font.Font(size=9)).pack()

        details_wrap = tk.Frame(card, bg="white")
        details_wrap.pack(fill="both", expand=True, padx=16)

        cols = ("Field", "Value")
        self.tree = ttk.Treeview(details_wrap, columns=cols, show="headings", height=18)
        self.tree.heading("Field", text="Field")
        self.tree.heading("Value", text="Value")
        self.tree.column("Field", width=300, anchor="c")
        self.tree.column("Value", width=380, anchor="c")

        vsb = ttk.Scrollbar(details_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Inline edit bindings
        self.tree.bind("<Double-1>", self._start_cell_edit)
        self.tree.bind("<Button-1>", self._maybe_finish_edit)

        # Calculate & AI Output (improved alignment)
        frm3 = tk.Frame(card, bg="white"); frm3.pack(pady=12)

        # Use an inner frame for horizontal alignment
        inner = tk.Frame(frm3, bg="white")
        inner.pack(anchor="center")

        # AI Predicted Probability label and value
        ai_prob_label = tk.Label(inner, text="AI Predicted Probability:", bg="white", fg=self.ENTRY_FG, font=font.Font(size=12))
        ai_prob_label.grid(row=0, column=0, sticky="w", padx=(0, 2))
        self.dpd_var = tk.StringVar()
        ai_prob_value = tk.Label(inner, textvariable=self.dpd_var, bg="white", fg=self.ENTRY_FG, font=font.Font(size=12), width=8, anchor="w")
        ai_prob_value.grid(row=0, column=1, sticky="w", padx=(2, 18), ipadx=10)

        # Credit Score label and value
        credit_label = tk.Label(inner, text="Credit Score:", bg="white", fg=self.ENTRY_FG, font=font.Font(size=12))
        credit_label.grid(row=0, column=2, sticky="w", padx=(0, 2))
        self.credit_var = tk.StringVar()
        credit_value = tk.Label(inner, textvariable=self.credit_var, bg="white", fg=self.ENTRY_FG, font=font.Font(size=12), width=8, anchor="w")
        credit_value.grid(row=0, column=3, sticky="w", padx=(2, 18), ipadx=10)

        # Button: Calculate Credit Score
        self.btn_calculate = tk.Button(inner, text="Calculate Credit Score",
                 command=self.calculate_score,
                 bg=self.GREEN, fg="black",
                 activebackground=self.GREEN, activeforeground="black",
                 font=font.Font(size=12), cursor="hand2")
        self._decorate(self.btn_calculate)
        self.btn_calculate.grid(row=0, column=4, sticky="w", ipadx=10, ipady=6, padx=(0, 0))

        # NEW: "Show breakdown" button (starts hidden)
        self.btn_breakdown = tk.Button(
            inner, text="Show\nBreakdown", command=self._show_breakdown_dialog,
            bg="#EEEEEE", fg="#000000", cursor="hand2"
        )
        self.btn_breakdown_is_packed = False

    # —— Fetch & map —— #
    def _coerce_csv_row(self, row: dict) -> dict:
        """Convert raw CSV values to model-friendly primitive types."""
        numeric_fields = {
            'Amount Finance', 'Downpayment', 'Downpayment %', 'Payment Terms', 'Price Gap',
            'Loan History', 'BAP Amount', 'Interest Rate', 'Claimed Income', 'EV Income', 'BV Income',
            'Claimed Tenure', 'EV Tenure', 'BV Tenure', 'DSCR', 'Age as of Contract',
            'Home Ownership', 'Residence Tenure', 'Loan Amort', 'Co-Borrowers', 'Borrower Type',
            'Contract Start Date'
        }
        coerced = {}
        for key, value in row.items():
            if value is None:
                coerced[key] = None
                continue
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    coerced[key] = None
                    continue
                if key in numeric_fields and value.lower() not in {'nan', 'none', 'null'}:
                    try:
                        cleaned = value.replace(',', '')
                        coerced[key] = float(cleaned)
                        continue
                    except ValueError:
                        pass
                if key == 'Contract Start Date':
                    try:
                        coerced[key] = pd.to_datetime(value, errors='raise').strftime('%Y-%m-%d')
                        continue
                    except Exception:
                        pass
            elif key == 'Contract Start Date' and value is not None and not (isinstance(value, float) and pd.isna(value)):
                try:
                    coerced[key] = pd.to_datetime(value).strftime('%Y-%m-%d')
                    continue
                except Exception:
                    pass
            coerced[key] = value
        return coerced

    def _get_local_dataset(self):
        """Load and cache local dataset from src/data for offline lookup."""
        cached = self.__dict__.get("_cached_local_df") if hasattr(self, "__dict__") else None
        if cached is not None:
            return cached
        candidates = [
            os.path.join(project_root, "src", "data", "Extracted New.csv"),
            os.path.join(project_root, "src", "data", "Extracted New.xlsx"),
            os.path.join(project_root, "src", "data", "May 12 Status Contracts.xlsx"),
            os.path.join(project_root, "src", "data", "May 12 Status Contracts.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                try:
                    df = read_csv_safe(p)
                    if not df.empty:
                        self._cached_local_df = df
                        return df
                except Exception:
                    pass
        return None

    def _find_row_by_party_and_app(self, df, party_no, app_no):
        """Find row in DataFrame matching Party No and Application No."""
        if df is None or df.empty:
            return None
        p_col = next((c for c in df.columns if str(c).strip().lower() in ['party no', 'party_no', 'partyno']), None)
        a_col = next((c for c in df.columns if str(c).strip().lower() in ['application no', 'application_no', 'applicationno', 'app no', 'app_no']), None)
        if not p_col or not a_col:
            return None
        try:
            p_match = pd.to_numeric(df[p_col], errors='coerce') == int(party_no)
            a_match = pd.to_numeric(df[a_col], errors='coerce') == int(app_no)
            matched = df[p_match & a_match]
            if not matched.empty:
                return matched.iloc[0].to_dict()
        except Exception:
            pass
        return None

    def fetch_customer(self):
        # Clear previous
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.dpd_var.set("")
        # clear credit score too
        if hasattr(self, "credit_var"):
            self.credit_var.set("")
        # Hide breakdown button if visible
        if hasattr(self, "btn_breakdown") and self.btn_breakdown_is_packed:
            self.btn_breakdown.grid_forget()
            self.btn_breakdown_is_packed = False
        # Show calculate button if hidden
        if hasattr(self, "btn_calculate"):
            self.btn_calculate.grid_forget()
            self.btn_calculate.grid(row=0, column=4, sticky="w", ipadx=10, ipady=6, padx=(0, 0))
        self.db_row = None
        self.ai_features = None
        self.contract_date_defaulted = False

        # Read optional Party No and Application No entered by user
        party_txt = self.ent_party.get().strip() if hasattr(self, "ent_party") else ""
        app_txt = self.ent_app.get().strip() if hasattr(self, "ent_app") else ""
        if party_txt.lower().startswith("party no"):
            party_txt = ""
        if app_txt.lower().startswith("application no"):
            app_txt = ""

        party_no = None
        application_no = None
        if party_txt and app_txt:
            try:
                party_no = int(party_txt)
                application_no = int(app_txt)
            except ValueError:
                pass

        # if uploaded data of user then, 
        if self.csv_mode.get():
            # Load from CSV / Excel
            data_dir = os.path.join(project_root, "src", "data")
            initial_dir = data_dir if os.path.exists(data_dir) else project_root
            csv_path = filedialog.askopenfilename(
                title="Select Data File",
                initialdir=initial_dir,
                filetypes=[
                    ("All supported data files (*.csv;*.xlsx;*.xls)", "*.csv;*.xlsx;*.xls"),
                    ("CSV files (*.csv)", "*.csv"),
                    ("Excel files (*.xlsx;*.xls)", "*.xlsx;*.xls"),
                    ("All files (*.*)", "*.*"),
                ],
            )
            if not csv_path:
                return
            try:
                df = read_csv_safe(csv_path)
                if df.empty:
                    return messagebox.showerror("File Error", "The file is empty.")

                # Check if user accidentally selected an export results file (e.g. test_results.csv)
                cols_lower = [str(c).strip().lower() for c in df.columns]
                if "predicted risk" in cols_lower and "credit score" in cols_lower:
                    return messagebox.showerror(
                        "Invalid File Selected",
                        f"The selected file '{os.path.basename(csv_path)}' is an output predictions report, not an input customer file.\n\n"
                        "Please select an input data file containing customer features, such as:\n"
                        "  • src/data/Extracted New.csv\n"
                        "  • test_data/SingleTest.csv"
                    )

                df = df.drop_duplicates()

                # If user entered Party No & App No, search for that specific record in the file
                row = None
                if party_no is not None and application_no is not None:
                    row = self._find_row_by_party_and_app(df, party_no, application_no)
                    if row is None:
                        messagebox.showwarning(
                            "Record Not Found in File",
                            f"No record matching Party No: {party_no} and Application No: {application_no} "
                            f"found in {os.path.basename(csv_path)}.\n\nUsing the first row in the file instead."
                        )

                if row is None:
                    if len(df) > 1:
                        messagebox.showinfo(
                            "Multiple Rows",
                            f"File contains {len(df)} rows. Using row 1 for inference.\n\n"
                            "Tip: You can enter Party No and Application No in the input fields above to select any specific customer."
                        )
                    row = df.iloc[0].to_dict()

                # --- Validate that the file has recognisable feature columns (direct or alias) ---
                csv_cols = set(str(c).strip() for c in df.columns)
                matched = csv_cols & set(self.ai.RAW_INPUT_FEATURES)
                normalized_preview = self._normalize_input_row(row)
                if not matched and not normalized_preview:
                    preview = ', '.join(sorted(csv_cols)[:8])
                    if len(csv_cols) > 8:
                        preview += '…'
                    return messagebox.showerror(
                        "File Error",
                        "The selected file has no recognised feature columns.\n\n"
                        "Expected columns such as:\n"
                        "  Amount Finance, Downpayment, DSCR, Interest Rate …\n\n"
                        f"Found: {preview}"
                    )

                self.db_row = self._coerce_csv_row(row)
            except Exception as e:
                return messagebox.showerror("File Error", f"Failed to load file: {e}")
        else:
            # Original DB fetch with automatic fallback to local dataset
            if not party_txt or not app_txt:
                return messagebox.showwarning("Missing", "Please enter Party No and Application No.")

            try:
                party_no = int(party_txt)
                application_no = int(app_txt)
            except ValueError:
                return messagebox.showerror("Invalid", "Party No and Application No must be integers.")

            row = None
            db_error = None
            try:
                row = fetch_ci_mock_crifv4(party_no, application_no)
            except Exception as e:
                db_error = e

            # If DB lookup failed or returned no row, fallback to local data in src/data/
            if not row:
                local_df = self._get_local_dataset()
                if local_df is not None:
                    row = self._find_row_by_party_and_app(local_df, party_no, application_no)
                    if row:
                        print(f"[Local Dataset] Loaded Party No: {party_no}, App No: {application_no} from local data file.")

            if not row:
                if db_error:
                    return messagebox.showerror(
                        "Record Not Found",
                        f"Database is unavailable ({db_error}) and record for Party No: {party_no}, "
                        f"Application No: {application_no} was not found in local data (src/data/Extracted New.csv)."
                    )
                return messagebox.showinfo("No Data", f"No matching record found for Party No: {party_no}, Application No: {application_no}.")

            self.db_row = self._coerce_csv_row(row)

        self.ai_features = self._row_to_ai_features(self.db_row)

        for feat_name in self.ai.RAW_INPUT_FEATURES:
            val = self.ai_features.get(feat_name, "")
            if val is None or val == "":
                val = "Missing"
            self.tree.insert("", "end", values=(feat_name, str(val)))

    # —— Inline editing helpers —— #
    def _strip_defaulted_note(self, v: str) -> str:
        if not isinstance(v, str):
            return v
        note_idx = v.find(" (defaulted from missing)")
        return v[:note_idx] if note_idx != -1 else v
    
    def _start_cell_edit(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self.tree.identify_column(event.x)
        if col != "#2":
            return
        row = self.tree.identify_row(event.y)
        if not row:
            return
        self._finish_edit(commit=True)
        x, y, w, h = self.tree.bbox(row, col)
        value = self.tree.set(row, "Value")
        value = self._strip_defaulted_note(value)
        self._edit_entry = tk.Entry(self.tree)
        self._edit_entry.insert(0, value)
        self._edit_entry.select_range(0, tk.END)
        self._edit_entry.focus()
        self._edit_entry.place(x=x, y=y, width=w, height=h)
        self._editing["item"] = row
        self._editing["column"] = "Value"
        self._edit_entry.bind("<Return>", lambda e: self._finish_edit(commit=True))
        self._edit_entry.bind("<Escape>", lambda e: self._finish_edit(commit=False))
        self._edit_entry.bind("<FocusOut>", lambda e: self._finish_edit(commit=True))

    def _maybe_finish_edit(self, _event):
        self._finish_edit(commit=True)

    def _finish_edit(self, commit=True):
        if self._edit_entry is None:
            return
        row = self._editing.get("item")
        col = self._editing.get("column")
        new_val = self._edit_entry.get().strip() if commit else None
        self._edit_entry.destroy()
        self._edit_entry = None
        self._editing = {"item": None, "column": None}
        if not commit or row is None or col is None:
            return
        self.tree.set(row, col, new_val if new_val != "" else "Missing")
        feat_name = self.tree.set(row, "Field")
        clean_val = self._strip_defaulted_note(new_val)
        if feat_name in self.ai.categorical_cols:
            self.ai_features[feat_name] = (clean_val if clean_val else "Missing")
        else:
            if isinstance(clean_val, str):
                clean_val = clean_val.replace(",", "")
            self.ai_features[feat_name] = clean_val

    def _sync_tree_to_features(self):
        if not self.ai_features:
            return
        for item in self.tree.get_children():
            feat, val = self.tree.item(item, "values")
            val = self._strip_defaulted_note(val)
            if feat in self.ai.RAW_INPUT_FEATURES:
                self.ai_features[feat] = val if val and str(val).lower() != "missing" else "Missing"
            else:
                self.ai_features[feat] = val if val and str(val).lower() != "missing" else "Missing"

    def _normalize_input_row(self, row: dict) -> dict:
        if row is None:
            return {}
        source = {}
        for key, value in dict(row).items():
            if value is None:
                continue
            key_text = str(key).strip()
            if key_text == "":
                continue
            value_text = str(value).strip()
            if value_text.lower() in {"nan", "none", "null"}:
                continue
            source[key_text] = value

        aliases = {
            'Amount Finance': ['Amount Finance', 'Amount Financed - Amount', 'AmountFinance'],
            'Downpayment': ['Downpayment', 'Downpayment - Amount', 'DP'],
            'Downpayment %': ['Downpayment %', 'DownpaymentPercent', 'DP %'],
            'Payment Terms': ['Payment Terms', 'Payment Terms - Number', 'Terms', 'Term in Months', 'Loan Term'],
            'Asset': ['Asset'],
            'Manufacturer': ['Manufacturer', 'Unit Brand', 'Unit Brand   - Scoring Code'],
            'Type of Unit': ['Type of Unit', 'Type of unit  - Scoring Code'],
            'Price Gap': ['Price Gap', 'Price Gap - Scoring Code'],
            'Loan History': ['Loan History', 'Loan History - Scoring Code'],
            'BAP Type': ['BAP Type', 'BAP / CIC Findings - Scoring Code'],
            'BAP Amount': ['BAP Amount', 'BAPScore'],
            'Interest Rate': ['Interest Rate'],
            'Claimed Income': ['Claimed Income', 'INCOME - Amount', 'TotalNetCombined', 'NetMonthly'],
            'EV Income': ['EV Income'],
            'BV Income': ['BV Income'],
            'Employment Type': ['Employment Type', 'Employment Type  - type', 'EmpType', 'EmploymentCategory'],
            'Claimed Tenure': ['Claimed Tenure', 'Employment / Business Tenure  - Scoring Code', 'TenureCategoryScore'],
            'EV Tenure': ['EV Tenure'],
            'BV Tenure': ['BV Tenure'],
            'DSCR': ['DSCR', 'DSCR - %'],
            'Age as of Contract': ['Age as of Contract', 'Age', 'AGE'],
            'Home Ownership': ['Home Ownership', 'Home Ownership - TYPE', 'HomeOwner'],
            'Residence Tenure': ['Residence Tenure', 'Residence Tenure  - Scoring Code', 'ResidenceScore'],
            'Citizenship': ['Citizenship', 'Citizenship  - type', 'Nationality', 'CBNationality'],
            'Dealer Name': ['Dealer Name', 'Type of dealer  - Scoring Code', 'DealerScore'],
            'Submitted ID': ['Submitted ID', 'Submitted ID   - type', 'IDSubmit'],
            'Marital Status': ['Marital Status', 'Marital Status - Scoring Code', 'CivilScore'],
            'Loan Amort': ['Loan Amort', 'LOAN AMORT', 'EMIFunction'],
            'Product Type': ['Product Type', 'Type', 'LoanPurpose'],
            'Mode of Payment': ['Mode of Payment'],
            'Co-Borrowers': ['Co-Borrowers', 'Co-Borrowers', 'CBCount'],
            'Borrower Type': ['Borrower Type'],
            'Contract Start Date': ['Contract Start Date', 'ContractStartDate'],
        }

        normalized = {}
        for canonical, values in aliases.items():
            for candidate in values:
                if candidate in source:
                    normalized[canonical] = source[candidate]
                    break

        for key in source:
            if key in AIModel.RAW_INPUT_FEATURES and key not in normalized:
                normalized[key] = source[key]

        return normalized

    # —— Mapping row -> features —— #
    def _row_to_ai_features(self, r: dict) -> dict:
        if r is None:
            return {}

        source = self._normalize_input_row(r)
        normalized = {}
        for key in AIModel.RAW_INPUT_FEATURES:
            value = source.get(key)
            if value is None or str(value).strip() == "":
                normalized[key] = None
                continue

            if key in {'Amount Finance', 'Downpayment', 'Downpayment %', 'Payment Terms', 'BAP Amount', 'Interest Rate', 'Claimed Income', 'EV Income', 'BV Income', 'Claimed Tenure', 'EV Tenure', 'BV Tenure', 'DSCR', 'Age as of Contract', 'Residence Tenure', 'Loan Amort'}:
                try:
                    normalized[key] = float(str(value).replace(',', ''))
                except Exception:
                    normalized[key] = value
            elif key in {'Product Type', 'Mode of Payment', 'Asset', 'Manufacturer', 'Type of Unit', 'Loan History', 'BAP Type', 'Employment Type', 'Home Ownership', 'Citizenship', 'Dealer Name', 'Submitted ID', 'Marital Status', 'Co-Borrowers', 'Borrower Type'}:
                normalized[key] = str(value).strip()
            else:
                normalized[key] = value

        if 'Co-Borrowers' in normalized and normalized['Co-Borrowers'] in (None, '', 'None'):
            normalized['Co-Borrowers'] = 'Missing'
        if 'Borrower Type' in normalized and normalized['Borrower Type'] in (None, '', 'None'):
            normalized['Borrower Type'] = 'Principal'
        if 'Product Type' in normalized and normalized['Product Type'] in (None, '', 'None'):
            normalized['Product Type'] = 'Missing'
        if 'Mode of Payment' in normalized and normalized['Mode of Payment'] in (None, '', 'None'):
            normalized['Mode of Payment'] = 'Unknown'

        return normalized
    
    # —— Predict —— #
    def _predict_with_current(self):
        self._sync_tree_to_features()
        feats = self.ai_features or self._row_to_ai_features(self.db_row)
        ai_pred = self.ai.predict_from_features(feats)
        self.dpd_var.set(f"{ai_pred:.2f}")

        # NOTE: compute & show Credit Score = Probability of being category 0 * 100
        credit_score = (1 - ai_pred) * 100.0
        if hasattr(self, "credit_var"):
            self.credit_var.set(f"{credit_score:.2f}")

        # Audit log the prediction
        try:
            party_no = self.db_row.get("Party No") if self.db_row else None
            app_no = self.db_row.get("Application No") if self.db_row else None
            threshold = getattr(self.ai, "decision_threshold", 0.5)
            log_prediction(party_no=party_no, app_no=app_no, probability=ai_pred,
                           credit_score=credit_score, threshold=threshold)
            logger.info("Scored customer party=%s, app=%s: prob=%.4f, score=%.2f",
                        party_no, app_no, ai_pred, credit_score)
        except Exception as log_err:
            logger.warning("Failed to log prediction: %s", log_err)

        # Hide calculate button after prediction
        if hasattr(self, "btn_calculate"):
            self.btn_calculate.grid_forget()

        # Reveal "Show breakdown" button on first successful prediction
        if hasattr(self, "btn_breakdown") and not self.btn_breakdown_is_packed:
            self.btn_breakdown.grid(row=0, column=4, sticky="w", ipadx=10, ipady=6, padx=(0, 0))
            self.btn_breakdown_is_packed = True

        messagebox.showinfo("AI Prediction", "Prediction complete.\n\nNote: Probability reflects risk of ANY DPD occurrence (even small overdue amounts are captured).")

    def _show_breakdown_dialog(self):
        if not self.ai_features:
            return messagebox.showwarning("Breakdown", "Please calculate a prediction first.")
        try:
            # Get SHAP breakdown from the model
            result = self.ai.breakdown_for_features(self.ai_features)
        except Exception as e:
            return messagebox.showerror("Breakdown Error", str(e))

        top = tk.Toplevel(self)
        top.title("Prediction Breakdown")
        top.configure(bg="white")
        top.geometry("700x560")
        top.minsize(640, 420)

        # Summary header
        header = tk.Frame(top, bg="white")
        header.pack(fill="x", padx=12, pady=(12, 6))

        def lab(txt, val):
            frm = tk.Frame(header, bg="white"); frm.pack(side="left", padx=14)
            tk.Label(frm, text=txt, bg="white", fg="#333", font=font.Font(size=10)).pack(anchor="w")
            tk.Label(frm, text=val, bg="white", fg="#000", font=font.Font(size=12, weight="bold")).pack(anchor="w")

        lab("Base value", f"{result['base_value']:.4f}")
        lab("Predicted Any-DPD Risk", f"{result['prediction']}")  # Any DPD detection (DPD > 0)
        lab("Base + ΣSHAP", f"{result['reconstructed']:.4f}")

        # Table with contributions
        wrap = tk.Frame(top, bg="white"); wrap.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        cols = ("Feature", "Value", "Impact (SHAP)", "|Impact|")
        tree = ttk.Treeview(wrap, columns=cols, show="headings", height=18)
        for c, w, anchor in (
            ("Feature", 300, "w"),
            ("Value", 200, "w"),
            ("Impact (SHAP)", 120, "e"),
            ("|Impact|", 100, "e"),
        ):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor=anchor)

        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscroll=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Insert rows (already sorted by abs impact)
        for row in result["rows"]:
            tree.insert("", "end", values=(
                row["feature"],
                str(row["value"]),
                f"{row['shap']:.6f}",
                f"{row['abs_shap']:.6f}",
            ))

        # Close shortcut
        top.bind("<Escape>", lambda _e: top.destroy())

    def calculate_score(self):
        if not self.db_row:
            return messagebox.showwarning("No Data", "Please fetch a customer first.")
        try:
            self._predict_with_current()
        except Exception as e:
            return messagebox.showerror("Prediction Error", f"Failed to predict: {e}")

    # ---- database settings dialog ----
    def _open_db_settings(self):
        """Display a separate window where user may view/edit DB connection info."""
        cfg = load_config()
        top = tk.Toplevel(self)
        top.title("Database Settings")
        top.geometry("460x320")
        top.configure(bg="white")
        top.resizable(False, False)

        # header label
        hdr = tk.Label(top, text="Database Connection", bg="white",
                       fg=self.GREEN, font=font.Font(size=16, weight="bold"))
        hdr.pack(pady=(16, 8))

        entries = {}
        for label_text, key in (
            ("Server", "server"),
            ("Database", "database"),
            ("User", "user"),
            ("Password", "password"),
            ("Driver", "driver"),
        ):
            frm = tk.Frame(top, bg="white")
            frm.pack(fill="x", padx=24, pady=6)
            tk.Label(frm, text=label_text + ":", bg="white", width=12, anchor="w",
                     font=font.Font(size=10)).pack(side="left")
            ent = tk.Entry(frm, width=34)
            ent.pack(side="right", fill="x", expand=True)
            ent.insert(0, cfg.get(key, ""))
            if key == "password":
                ent.config(show="*")
            entries[key] = ent

        # buttons frame
        btnfrm = tk.Frame(top, bg="white")
        btnfrm.pack(pady=12)
        def save():
            newcfg = {k: e.get().strip() for k, e in entries.items()}
            try:
                save_config(newcfg)
                messagebox.showinfo("Settings", "Database configuration saved.")
                top.destroy()
            except Exception as exc:
                messagebox.showerror("Save Error", f"Could not save config:\n{exc}")

        btn_save = tk.Button(btnfrm, text="Save", command=save,
                        bg=self.GREEN, fg="black",
                        activebackground=self.GREEN, activeforeground="black",
                        font=font.Font(size=12), cursor="hand2")
        btn_save.pack(side="left", ipadx=10, ipady=4, padx=10)
        self._decorate(btn_save)
        btn_cancel = tk.Button(btnfrm, text="Cancel", command=top.destroy,
                        bg="#AAA", fg="black",
                        activebackground="#AAA", activeforeground="black",
                        font=font.Font(size=12), cursor="hand2")
        btn_cancel.pack(side="left", ipadx=10, ipady=4, padx=10)
        self._decorate(btn_cancel)

    # —— 'Multiple' CREDIT SCORING SCREEN —— #
    def _batch_processing(self):
        self.geometry("950x850")
        if hasattr(self, 'scoring_frame'):
            self.scoring_frame.place_forget()
        if hasattr(self, 'multiple_frame'):
            self.multiple_frame.place_forget()

        self.multiple_frame = tk.Frame(self, bg=self.BG_GRAY)
        self.multiple_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        card = tk.Frame(self.multiple_frame, bg='white')
        card.place(relx=0.5, rely=0.5, anchor='center', width=900, height=780)

        self._batch_file_path = tk.StringVar()  # reacts to file selection
        self._batch_df = None                   # raw uploaded dataframe
        self._batch_results = []                # list of result dicts after compute

        tk.Label(card, image=self.logo, bg="white").pack(pady=(20, 5))

        # -- Back to previous window -- #
        com_one_btn = tk.Button(card, text="«", command=self._show_scoring,
                                bg='white', fg='black', bd=0,
                                font=font.Font(size=24), cursor="hand2")
        com_one_btn.place(relx=0.0, rely=0.0, anchor='nw', x=8, y=0)

        # gear/settings button
        gear_btn = tk.Button(card, text="⚙", command=self._open_db_settings,
                             bg='white', fg="black", bd=0,
                             font=font.Font(size=14), cursor="hand2")
        gear_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

        tk.Label(card, text="Credit Score Calculator", bg="white", fg=self.GREEN,
                 font=font.Font(size=18, weight="bold")).pack(pady=(0, 15))

        # -- Upload file -- #
        frm = tk.Frame(card, bg="white")
        frm.pack(pady=(0, 10), padx=16, fill="x")

        self._batch_path_entry = PlaceholderEntry(frm, placeholder="Upload required file path .csv",
                                                  fg=self.ENTRY_FG, bg=self.ENTRY_BG,
                                                  insertbackground=self.ENTRY_FG, font=font.Font(size=12), width=38,
                                                  textvariable=self._batch_file_path)
        
        self._batch_path_entry.pack(side="left", ipady=6, padx=(0, 8), fill="x", expand=True)
        self._decorate(self._batch_path_entry)

        self.btn_browse = tk.Button(frm, text="Browse", command=self._select_batch_file,
                               bg=self.GREEN, fg="black", activebackground='lightgreen',
                               activeforeground="black", font=font.Font(size=12), cursor="hand2")
        self.btn_browse.pack(side="left", padx=(0, 8), ipady=6, ipadx=10)
        self._decorate(self.btn_browse)

        self.btn_upload = tk.Button(frm, text="Upload", command=self._load_batch_file,
                               bg=self.GREEN, fg="black", activebackground='lightgreen',
                               activeforeground="black", font=font.Font(size=12), cursor="hand2")
        self.btn_upload.pack(side="left", ipady=6, ipadx=10)
        self._decorate(self.btn_upload)

        # Store the frame to hide later
        self.upload_frame = frm

        # Description label below the upload frame
        desc_label = tk.Label(card, text="Upload a CSV file with required columns: 'Application No' and 'Party No'.\nOther columns should match the model's feature names for accurate predictions.",
                              bg="white", fg="#555", font=font.Font(size=10), justify="center")
        desc_label.pack(anchor="center", pady=(0, 10))
        self.desc_label = desc_label

        self._batch_table_frame = tk.Frame(card, bg='white')
        self._batch_table_frame.pack(fill='both', expand=True, padx=16, pady=(6, 10))
    
    def _batch_featch_data(self):
        df = self._batch_df
        if df is None or df.empty:
            return messagebox.showwarning("Missing Data", "Please upload a CSV file first.")

        if "Application No" not in df.columns or "Party No" not in df.columns:
            return messagebox.showerror("Invalid Columns", "Batch file must include 'Application No' and 'Party No'.")

        self._batch_results = []
        
        # Try connection to database
        conn = None
        try:
            # conn = get_connection_local()
            conn = get_connection()
            if conn is None:
                raise ValueError("Connection unavailable")
        except Exception as e:
            conn = None
            messagebox.showwarning("DB Warning", f"DB connection failed; using fallback mode: {e}")

        # Choose processing method based on connection availability
        if conn is not None:
            self._batch_featch_data_db(conn)
        else:
            # Fallback: use uploaded row data directly (no DB lookup needed)
            self._batch_featch_data_fallback()

        # -- Process status popup window
        cnt = len([r for r in self._batch_results if r.get("Predicted DPD (Any)") is not None])
        messagebox.showinfo("Batch Processing", f"Processed {cnt} / {len(self._batch_results)} records.")
        self._render_batch_results_table()

    def _batch_featch_data_db(self, conn):
        """Process batch data using database connection."""
        df = self._batch_df
        
        for _, row in df.iterrows():
            try:
                app_no = int(row["Application No"])
                party_no = int(row["Party No"])
            except Exception:
                continue

            db_row = None
            result = {
                "Application No": app_no,
                "Party No": party_no,
                "AI Predicted Probability": None,
                "Predicted DPD (Any)": None,
                "Credit Score": None,
                "Base Value": None,
                "Reconstructed": None,
                "Status": "OK",
                "source_row": None,
            }

            # Fetch from database
            try:
                db_row = fetch_ci_mock_crifv4(party_no=party_no, application_no=app_no, conn=conn)
            except Exception as e:
                result["Status"] = f"DB Fetch Error: {str(e)}"
                self._batch_results.append(result)
                continue

            if db_row is None:
                result["Status"] = "No matching record in database"
                self._batch_results.append(result)
                continue

            # Process prediction
            self._process_batch_prediction(result, db_row)

    def _batch_featch_data_fallback(self):
        """Process batch data using sample_data.csv fallback and uploaded row data."""
        df = self._batch_df
        
        # -- Fallback to sample_data.csv for testing 
        sample_df = None
        try:
            sample_df = pd.read_csv(resource_path("test_data/sample_data.csv"))
        except Exception:
            sample_df = None

        for _, row in df.iterrows():
            try:
                app_no = int(row["Application No"])
                party_no = int(row["Party No"])
            except Exception:
                continue

            db_row = None
            result = {
                "Application No": app_no,
                "Party No": party_no,
                "AI Predicted Probability": None,
                "Predicted DPD (Any)": None,
                "Credit Score": None,
                "Base Value": None,
                "Reconstructed": None,
                "Status": "OK",
                "source_row": None,
            }

            if sample_df is not None and "Application No" in sample_df.columns and "Party No" in sample_df.columns:
                match = sample_df[(sample_df["Application No"] == app_no) & (sample_df["Party No"] == party_no)]
                if len(match) > 0:
                    db_row = match.iloc[0].to_dict()
            
            # Last resort: use uploaded row data
            if db_row is None:
                db_row = row.to_dict()

            # Process prediction
            self._process_batch_prediction(result, db_row)

    def _process_batch_prediction(self, result, db_row):
        """Helper method to process prediction for a single batch row."""
        result["source_row"] = db_row
        normalized = self._normalize_input_row(db_row)
        feats = self._row_to_ai_features(normalized)
        
        try:
            ai_pred = self.ai.predict_from_features(feats)
            # Classify as default (1) or good (0) using the model's calibrated threshold
            threshold = getattr(self.ai, 'decision_threshold', 0.5)
            predicted_dpd = int(ai_pred >= threshold)  # 1 = any DPD likely, 0 = clean
            result["AI Predicted Probability"] = round(ai_pred, 6)
            result["Predicted DPD (Any)"] = predicted_dpd
            result["Credit Score"] = round((1 - ai_pred) * 100.0, 2)
        except Exception as e:
            result["Status"] = f"Predict error: {e}"
            self._batch_results.append(result)
            return

        try:
            breakdown = self.ai.breakdown_for_features(feats)
            result["Base Value"] = round(breakdown.get("base_value", 0.0), 6)
            result["Reconstructed"] = round(breakdown.get("reconstructed", 0.0), 6)
        except Exception:
            result["Base Value"] = "N/A"
            result["Reconstructed"] = "N/A"

        self._batch_results.append(result)


    def _render_batch_results_table(self):
        if not self._batch_results:
            return

        for child in self._batch_table_frame.winfo_children():
            child.destroy()

        toolbar = tk.Frame(self._batch_table_frame, bg='white')
        toolbar.pack(fill='x', pady=(0, 6))

        # Export format selection variable
        if not hasattr(self, '_export_format'):
            self._export_format = tk.StringVar(value="xlsx")

        # Radio buttons for export format (side by side)
        radio_frame = tk.Frame(toolbar, bg='white')
        radio_frame.pack(side='right', padx=(0, 10))

        tk.Radiobutton(radio_frame, text='XLSX', variable=self._export_format, value='xlsx',
                       bg='white', fg='black').pack(side='left', padx=5)
        tk.Radiobutton(radio_frame, text='CSV', variable=self._export_format, value='csv',
                       bg='white', fg='black').pack(side='left', padx=5)

        # Save button
        def export_handler():
            fmt = self._export_format.get()
            if fmt == 'xlsx':
                self._export_batch_results_xlsx()
            else:
                self._export_batch_results()

        save_btn = tk.Button(radio_frame, text="Export Results", command=export_handler,
                             bg=self.GREEN, fg='black', cursor='hand2')
        save_btn.pack(side='left', padx=(10, 0), ipadx=8, ipady=4)
        self._decorate(save_btn)


        columns = ["Application No", "Party No", "AI Predicted Probability", "Predicted DPD (Any)", "Credit Score", "Base Value", "Reconstructed", "Status"]

        # Dedicated frame for Treeview and scrollbars
        table_frame = tk.Frame(self._batch_table_frame, bg='white')
        table_frame.pack(fill='both', expand=True)

        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=14)
        for c in columns:
            tree.heading(c, text=c)
            tree.column(c, anchor='center')  # wider columns to force horizontal scroll

        vsb = ttk.Scrollbar(table_frame, orient='vertical', command=tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient='horizontal', command=tree.xview)
        tree.configure(yscroll=vsb.set, xscroll=hsb.set)

        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        for r in self._batch_results:
            tree.insert('', 'end', values=[r.get(c, '') for c in columns])

        self._adjust_treeview_columns(tree, columns)

    def _adjust_treeview_columns(self, tree, columns, max_width=300):
        style = ttk.Style()
        font_name = style.lookup("Treeview", "font")
        font = tkFont.nametofont(font_name)

        for col in columns:
            max_w = font.measure(col)  # header width

            for item in tree.get_children():
                value = str(tree.set(item, col))
                max_w = max(max_w, font.measure(value))

            tree.column(col, width=min(max_w + 20, max_width))

    def _export_batch_results(self):
        if not self._batch_results:
            return messagebox.showwarning("No Results", "There are no batch results to export.")

        folder = filedialog.askdirectory(title="Select folder to save batch breakdown files")
        if not folder:
            return

        failed = 0
        saved_files = []
        for rec in self._batch_results:
            predicted = rec.get("Predicted DPD (Any)")
            if predicted is None:
                continue

            app_no = rec.get("Application No")
            party_no = rec.get("Party No")
            fname = f"{app_no}_{party_no}.csv"
            out_path = os.path.join(folder, fname)

            try:
                src = rec.get("source_row") or {}
                normalized_src = self._normalize_input_row(src)
                breakdown = self.ai.breakdown_for_features(normalized_src)

                header = [
                    ["Application No", app_no],
                    ["Party No", party_no],
                    ["Predicted DPD (Any)", predicted],  # 1=any DPD, 0=clean
                    ["Credit Score", rec.get("Credit Score")],
                    ["Base Value", rec.get("Base Value")],
                    ["Reconstructed", rec.get("Reconstructed")],
                    ["Status", rec.get("Status")],
                    [],
                ]

                detail_rows = []
                for row in breakdown.get("rows", []):
                    detail_rows.append({
                        "Feature": row.get("feature"),
                        "Value": row.get("value"),
                        "Impact (SHAP)": row.get("shap"),
                        "|Impact|": row.get("abs_shap"),
                    })

                with open(out_path, "w", newline="", encoding="utf-8") as f:
                    import csv
                    writer = csv.writer(f)
                    for h in header:
                        writer.writerow(h)
                    writer.writerow([])
                    writer.writerow(["Feature", "Value", "Impact (SHAP)", "|Impact|"])
                    for d in detail_rows:
                        writer.writerow([d["Feature"], d["Value"], d["Impact (SHAP)"], d["|Impact|"]])

                saved_files.append(out_path)
            except Exception as e:
                failed += 1

        if saved_files:
            messagebox.showinfo("Exported", f"Saved {len(saved_files)} breakdown files to:\n{folder}")
        if failed > 0:
            messagebox.showwarning("Partial Failure", f"{failed} files failed to save.")

    def _export_batch_results_xlsx(self):
        """Export batch results to a single XLSX file with summary sheet and detailed sheets."""
        if not self._batch_results:
            return messagebox.showwarning("No Results", "There are no batch results to export.")

        file_path = filedialog.asksaveasfilename(
            title="Save batch results as XLSX",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font, PatternFill, Alignment
            
            wb = Workbook()
            wb.remove(wb.active)  # Remove default sheet
            
            # Sheet 1: Summary table (from _render_batch_results_table)
            summary_sheet = wb.create_sheet("Summary", 0)
            columns = ["Application No", "Party No", "AI Predicted Probability", "Predicted DPD (Any)", "Credit Score", "Base Value", "Reconstructed", "Status"]
            
            # Write header with formatting
            header_fill = PatternFill(start_color="2E7D32", end_color="2E7D32", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for col_idx, col_name in enumerate(columns, 1):
                cell = summary_sheet.cell(row=1, column=col_idx)
                cell.value = col_name
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Write summary data rows
            for row_idx, rec in enumerate(self._batch_results, 2):
                if rec.get("Predicted DPD (Any)") is None:
                    continue
                
                summary_sheet.cell(row=row_idx, column=1).value = rec.get("Application No")
                summary_sheet.cell(row=row_idx, column=2).value = rec.get("Party No")
                summary_sheet.cell(row=row_idx, column=3).value = rec.get("AI Predicted Probability")
                summary_sheet.cell(row=row_idx, column=4).value = rec.get("Predicted DPD (Any)")
                summary_sheet.cell(row=row_idx, column=5).value = rec.get("Credit Score")
                summary_sheet.cell(row=row_idx, column=6).value = rec.get("Base Value")
                summary_sheet.cell(row=row_idx, column=7).value = rec.get("Reconstructed")
                summary_sheet.cell(row=row_idx, column=8).value = rec.get("Status")
            
            # Auto-adjust column widths for summary sheet
            for col in summary_sheet.columns:
                max_length = 0
                for cell in col:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                summary_sheet.column_dimensions[col[0].column_letter].width = min(max_length + 2, 50)
            
            # Sheets 2+: Detailed breakdown for each record
            for rec in self._batch_results:
                predicted = rec.get("Predicted DPD (Any)")
                if predicted is None:
                    continue
                
                app_no = rec.get("Application No")
                party_no = rec.get("Party No")
                sheet_name = f"Detail_{app_no}_{party_no}"[:31]  # Excel sheet name limit
                
                detail_sheet = wb.create_sheet(sheet_name)
                
                # Header info
                detail_sheet.cell(row=1, column=1).value = "Application No"
                detail_sheet.cell(row=1, column=2).value = app_no
                detail_sheet.cell(row=2, column=1).value = "Party No"
                detail_sheet.cell(row=2, column=2).value = party_no
                detail_sheet.cell(row=3, column=1).value = "Predicted DPD (Any)"
                detail_sheet.cell(row=3, column=2).value = predicted
                detail_sheet.cell(row=4, column=1).value = "Credit Score"
                detail_sheet.cell(row=4, column=2).value = rec.get("Credit Score")
                detail_sheet.cell(row=5, column=1).value = "Base Value"
                detail_sheet.cell(row=5, column=2).value = rec.get("Base Value")
                detail_sheet.cell(row=6, column=1).value = "Reconstructed"
                detail_sheet.cell(row=6, column=2).value = rec.get("Reconstructed")
                detail_sheet.cell(row=7, column=1).value = "Status"
                detail_sheet.cell(row=7, column=2).value = rec.get("Status")
                
                # Breakdown table
                src = rec.get("source_row") or {}
                normalized_src = self._normalize_input_row(src)
                breakdown = self.ai.breakdown_for_features(normalized_src)
                
                # Breakdown header
                detail_sheet.cell(row=9, column=1).value = "Feature"
                detail_sheet.cell(row=9, column=2).value = "Value"
                detail_sheet.cell(row=9, column=3).value = "Impact (SHAP)"
                detail_sheet.cell(row=9, column=4).value = "|Impact|"
                
                # Apply header formatting
                for col in range(1, 5):
                    cell = detail_sheet.cell(row=9, column=col)
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center")
                
                # Breakdown rows
                for row_idx, row in enumerate(breakdown.get("rows", []), 10):
                    detail_sheet.cell(row=row_idx, column=1).value = row.get("feature")
                    detail_sheet.cell(row=row_idx, column=2).value = row.get("value")
                    detail_sheet.cell(row=row_idx, column=3).value = row.get("shap")
                    detail_sheet.cell(row=row_idx, column=4).value = row.get("abs_shap")
                
                # Auto-adjust column widths
                detail_sheet.column_dimensions["A"].width = 40
                detail_sheet.column_dimensions["B"].width = 20
                detail_sheet.column_dimensions["C"].width = 15
                detail_sheet.column_dimensions["D"].width = 15
            
            wb.save(file_path)
            messagebox.showinfo("Export Complete", f"Results exported successfully to:\n{file_path}")
        
        except ImportError:
            messagebox.showerror("Missing Package", "openpyxl is required for XLSX export.\nInstall it using: pip install openpyxl")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")


    def _render_batch_table(self):
        if self._batch_df is None:
            return

        for child in self._batch_table_frame.winfo_children():
            child.destroy()

        cols = list(self._batch_df.columns)
        tree = ttk.Treeview(self._batch_table_frame, columns=cols, show='headings', height=12)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor='center')

        vsb = ttk.Scrollbar(self._batch_table_frame, orient='vertical', command=tree.yview)
        tree.configure(yscroll=vsb.set)

        tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        for _, row in self._batch_df.iterrows():
            tree.insert('', 'end', values=[str(row[c]) for c in cols])

    # -- Checking if headers present -- #        
    def check_file(self, df, required):
        if not required.issubset(set(df.columns)):
           return False 
        return True

    def _show_tooltip(self, event, text):
        self.tooltip_label.config(text=text)
        btn = event.widget
        x = btn.winfo_rootx() - self.winfo_rootx()
        y = btn.winfo_rooty() - self.winfo_rooty() - 30  # Position above the button
        self.tooltip_label.place(x=x, y=y)

    def _hide_tooltip(self, event):
        self.tooltip_label.place_forget()
    
if __name__ == "__main__":
    App().mainloop()