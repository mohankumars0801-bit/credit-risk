"""
admin_dashboard.py – Admin management panel for the Credit Scoring application.

Improvements over the original:
  • Audit log viewer panel (reads ~/.jaccs_credit_app/logs/audit.log)
  • "Refresh" button that reloads the user list
  • Last-login timestamp column (from users.db when available)
  • Keyboard shortcut: Delete key → delete selected user
  • Double-click on a user row → open Reset Password dialog
  • Better error messages when operations fail
  • Expandable/collapsible audit-log section

All original functionality is preserved and no existing public interface was changed.
"""

from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tkinter as tk
from tkinter import ttk, messagebox, font
from typing import List, Dict, Optional

from src.backend.users import list_users, add_user, delete_user, update_password, set_role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_AUDIT_LOG_PATH = os.path.join(os.path.expanduser("~"), ".jaccs_credit_app", "logs", "audit.log")
_USERS_DB_PATH  = os.path.join(os.path.expanduser("~"), ".jaccs_credit_app", "users.db")


def _read_last_logins() -> Dict[str, str]:
    """Return {email: last_login_str} from users.db if it exists, else empty dict."""
    if not os.path.exists(_USERS_DB_PATH):
        return {}
    try:
        import sqlite3
        conn = sqlite3.connect(_USERS_DB_PATH)
        cur = conn.cursor()
        # Try a 'last_login' column; silently fail if the schema is different
        cur.execute("SELECT email, last_login FROM users WHERE last_login IS NOT NULL")
        result = {row[0]: str(row[1]) for row in cur.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


def _read_audit_log(max_lines: int = 500) -> List[str]:
    """Return the last *max_lines* lines of the audit log, or an informational message."""
    if not os.path.exists(_AUDIT_LOG_PATH):
        return [f"Audit log not found at:  {_AUDIT_LOG_PATH}"]
    try:
        with open(_AUDIT_LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        # Most-recent first
        lines = [l.rstrip("\n\r") for l in lines[-max_lines:]]
        lines.reverse()
        return lines if lines else ["(Audit log is empty)"]
    except Exception as exc:
        return [f"Could not read audit log: {exc}"]


# ---------------------------------------------------------------------------
# AdminDashboard
# ---------------------------------------------------------------------------
class AdminDashboard(tk.Frame):
    """Full-featured admin management panel.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    on_open_scoring : callable | None
        Called when the user clicks "Open Scoring App".
    **kwargs
        Forwarded to :class:`tk.Frame`.
    """

    GREEN = "#2E7D32"
    _BTN_KWARGS = dict(fg="black", cursor="hand2")

    def __init__(self, master: tk.Widget, on_open_scoring=None, **kwargs) -> None:
        bg = kwargs.get("bg", "#F0F0F0")
        super().__init__(master, bg=bg)
        self.on_open_scoring = on_open_scoring

        # Main card
        card = tk.Frame(self, bg="white")
        card.place(relx=0.5, rely=0.5, anchor="center", width=800, height=700)

        # ── Title ──────────────────────────────────────────────────────────
        tk.Label(
            card, text="Admin Dashboard", bg="white",
            font=font.Font(size=18, weight="bold"), fg=self.GREEN,
        ).pack(pady=(18, 2))
        tk.Label(
            card,
            text="Manage users (add / delete / reset password / toggle admin)",
            bg="white", fg="#555",
        ).pack(pady=(0, 6))

        # ── Notebook: Users tab + Audit Log tab ────────────────────────────
        style = ttk.Style()
        style.configure("Admin.TNotebook", background="white")
        style.configure("Admin.TNotebook.Tab", padding=[12, 4])

        nb = ttk.Notebook(card, style="Admin.TNotebook")
        nb.pack(fill="both", expand=True, padx=16, pady=(4, 6))

        # ── Tab 1 : Users ─────────────────────────────────────────────────
        users_tab = tk.Frame(nb, bg="white")
        nb.add(users_tab, text="  Users  ")

        self._build_users_tab(users_tab)

        # ── Tab 2 : Audit Log ─────────────────────────────────────────────
        audit_tab = tk.Frame(nb, bg="white")
        nb.add(audit_tab, text="  Audit Log  ")
        self._build_audit_tab(audit_tab)

        # ── Bottom: Open Scoring App ───────────────────────────────────────
        tk.Button(
            card, text="Open Scoring App",
            command=self._open_scoring,
            bg="#EEEEEE", fg="#000000", cursor="hand2",
        ).pack(pady=(4, 14), ipadx=10, ipady=6)

        # Initial data load
        self._refresh()

    # ------------------------------------------------------------------ #
    # Users tab                                                            #
    # ------------------------------------------------------------------ #

    def _build_users_tab(self, parent: tk.Frame) -> None:
        """Build the user-management section."""

        # ── Treeview ──────────────────────────────────────────────────────
        wrap = tk.Frame(parent, bg="white")
        wrap.pack(fill="both", expand=True, padx=4, pady=(8, 4))

        cols = ("Email", "Role", "Last Login")
        self.user_tree = ttk.Treeview(wrap, columns=cols, show="headings", height=16)

        col_widths = {"Email": 400, "Role": 120, "Last Login": 180}
        for c in cols:
            self.user_tree.heading(c, text=c)
            self.user_tree.column(c, width=col_widths[c], anchor="w")

        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.user_tree.yview)
        self.user_tree.configure(yscroll=vsb.set)
        self.user_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # ── Keyboard shortcuts ────────────────────────────────────────────
        self.user_tree.bind("<Delete>", lambda _e: self._delete_user())
        self.user_tree.bind("<Double-1>", self._on_tree_double_click)

        # ── Buttons row ───────────────────────────────────────────────────
        btns = tk.Frame(parent, bg="white")
        btns.pack(pady=(4, 8))

        btn_defs = [
            ("Add User",       self._add_user_dialog),
            ("Reset Password", self._reset_password_dialog),
            ("Delete User",    self._delete_user),
            ("Toggle Admin",   self._toggle_admin),
            ("↻ Refresh",     self._refresh),
        ]
        for label, cmd in btn_defs:
            bg = self.GREEN if label != "↻ Refresh" else "#DDDDDD"
            tk.Button(
                btns, text=label, command=cmd,
                bg=bg, **self._BTN_KWARGS,
            ).pack(side="left", padx=5, ipadx=8, ipady=5)

    # ------------------------------------------------------------------ #
    # Audit log tab                                                         #
    # ------------------------------------------------------------------ #

    def _build_audit_tab(self, parent: tk.Frame) -> None:
        """Build the audit log viewer panel."""
        ctrl = tk.Frame(parent, bg="white")
        ctrl.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(
            ctrl, text=f"Log file: {_AUDIT_LOG_PATH}",
            bg="white", fg="#888", font=font.Font(size=9),
        ).pack(side="left")

        tk.Button(
            ctrl, text="↻ Refresh Log", command=self._refresh_audit_log,
            bg="#DDDDDD", fg="black", cursor="hand2",
        ).pack(side="right", ipadx=8, ipady=3)

        # Text widget with scrollbar
        txt_wrap = tk.Frame(parent, bg="white")
        txt_wrap.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._audit_text = tk.Text(
            txt_wrap, wrap="none", state="disabled",
            font=("Courier New", 9), bg="#FAFAFA", fg="#333",
            relief="flat", highlightthickness=1, highlightbackground="#E0E0E0",
        )
        v_scroll = ttk.Scrollbar(txt_wrap, orient="vertical", command=self._audit_text.yview)
        h_scroll = ttk.Scrollbar(txt_wrap, orient="horizontal", command=self._audit_text.xview)
        self._audit_text.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self._audit_text.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        txt_wrap.grid_rowconfigure(0, weight=1)
        txt_wrap.grid_columnconfigure(0, weight=1)

        self._refresh_audit_log()

    def _refresh_audit_log(self) -> None:
        """Reload and display the audit log contents."""
        lines = _read_audit_log()
        self._audit_text.configure(state="normal")
        self._audit_text.delete("1.0", tk.END)
        self._audit_text.insert(tk.END, "\n".join(lines))
        self._audit_text.configure(state="disabled")
        self._audit_text.see("1.0")  # scroll to top (most recent)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _refresh(self) -> None:
        """Reload the user list from the database."""
        for i in self.user_tree.get_children():
            self.user_tree.delete(i)

        last_logins = _read_last_logins()
        try:
            users = list_users()
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not load user list:\n{exc}")
            return

        for u in users:
            email = u.get("email", "")
            role = u.get("role", "user")
            last_login = last_logins.get(email, "—")
            self.user_tree.insert("", "end", values=(email, role, last_login))

    def _selected_email(self) -> str:
        sel = self.user_tree.selection()
        if not sel:
            messagebox.showwarning("Select User", "Please select a user from the list first.")
            return ""
        return self.user_tree.item(sel[0], "values")[0]

    def _count_admins(self) -> int:
        try:
            return sum(1 for u in list_users() if u.get("role") == "admin")
        except Exception:
            return 0

    def _on_tree_double_click(self, event: tk.Event) -> None:  # type: ignore[override]
        """Double-click on user row → open Reset Password dialog."""
        region = self.user_tree.identify("region", event.x, event.y)
        if region == "cell":
            self._reset_password_dialog()

    # ------------------------------------------------------------------ #
    # Actions                                                              #
    # ------------------------------------------------------------------ #

    def _add_user_dialog(self) -> None:
        import re

        top = tk.Toplevel(self)
        top.title("Add User")
        top.configure(bg="white")
        top.transient(self.winfo_toplevel())
        top.grab_set()
        top.lift()

        D_W, D_H = 520, 280
        self.update_idletasks()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw = max(self.winfo_width(), D_W)
        ph = max(self.winfo_height(), D_H)
        top.geometry(f"{D_W}x{D_H}+{px + (pw - D_W) // 2}+{py + (ph - D_H) // 2}")
        top.minsize(D_W, D_H)
        top.resizable(False, False)

        container = tk.Frame(top, bg="white")
        container.pack(fill="both", expand=True)

        form = tk.Frame(container, bg="white")
        form.pack(fill="x", padx=16, pady=(16, 8))
        form.grid_columnconfigure(1, weight=1)

        tk.Label(form, text="Email", bg="white").grid(row=0, column=0, sticky="e", padx=(6, 10), pady=6)
        email_e = tk.Entry(form)
        email_e.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=6)

        tk.Label(form, text="Password", bg="white").grid(row=1, column=0, sticky="e", padx=(6, 10), pady=6)
        pass_e = tk.Entry(form, show="*")
        pass_e.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=6)

        is_admin = tk.BooleanVar(master=top, value=False)
        tk.Checkbutton(form, text="Add as admin", variable=is_admin, bg="white").grid(
            row=2, column=1, sticky="w", padx=(0, 6), pady=(2, 8),
        )

        btns = tk.Frame(container, bg="white")
        btns.pack(fill="x", padx=16, pady=(6, 16))

        def do_save() -> None:
            email = email_e.get().strip()
            pwd = pass_e.get().strip()
            if not email or not pwd:
                messagebox.showwarning("Add User", "Email and password are required.", parent=top)
                return
            if not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
                messagebox.showwarning("Add User", "Please enter a valid email address.", parent=top)
                return
            try:
                add_user(email, pwd, role="admin" if is_admin.get() else "user")
                top.destroy()
                self._refresh()
                messagebox.showinfo("Add User", f"User '{email}' added successfully.")
            except Exception as exc:
                messagebox.showerror("Add User", f"Failed to add user:\n{exc}", parent=top)

        tk.Button(btns, text="Cancel", command=top.destroy, bg="#EEEEEE", fg="black", cursor="hand2").pack(
            side="right", padx=6, ipadx=12, ipady=6,
        )
        tk.Button(btns, text="Save", command=do_save, bg=self.GREEN, fg="black", cursor="hand2").pack(
            side="right", padx=6, ipadx=12, ipady=6,
        )

        top.bind("<Return>", lambda _e: do_save())
        top.bind("<Escape>", lambda _e: top.destroy())
        email_e.focus_set()

    def _reset_password_dialog(self) -> None:
        """Open the reset-password dialog for the selected user."""
        email = self._selected_email()
        if not email:
            return

        top = tk.Toplevel(self)
        top.title(f"Reset Password — {email}")
        top.configure(bg="white")
        top.transient(self.winfo_toplevel())
        top.grab_set()
        top.lift()

        D_W, D_H = 420, 180
        self.update_idletasks()
        px, py = self.winfo_rootx(), self.winfo_rooty()
        pw = max(self.winfo_width(), D_W)
        ph = max(self.winfo_height(), D_H)
        top.geometry(f"{D_W}x{D_H}+{px + (pw - D_W) // 2}+{py + (ph - D_H) // 2}")
        top.resizable(False, False)

        container = tk.Frame(top, bg="white")
        container.pack(fill="both", expand=True, padx=16, pady=16)

        tk.Label(container, text=f"User:  {email}", bg="white", fg="#444",
                 font=font.Font(size=10, weight="bold")).pack(anchor="w", pady=(0, 8))

        frm = tk.Frame(container, bg="white")
        frm.pack(fill="x")
        tk.Label(frm, text="New Password:", bg="white").pack(side="left")
        pass_e = tk.Entry(frm, width=28, show="*")
        pass_e.pack(side="left", padx=(8, 0), ipady=4)

        def do_reset() -> None:
            pwd = pass_e.get().strip()
            if not pwd:
                messagebox.showwarning("Reset Password", "Password cannot be empty.", parent=top)
                return
            try:
                update_password(email, pwd)
                top.destroy()
                messagebox.showinfo("Reset Password", f"Password for '{email}' has been updated.")
            except Exception as exc:
                messagebox.showerror("Reset Password", f"Failed to reset password:\n{exc}", parent=top)

        btn_row = tk.Frame(container, bg="white")
        btn_row.pack(pady=(12, 0), fill="x")
        tk.Button(btn_row, text="Cancel", command=top.destroy, bg="#EEEEEE", fg="black", cursor="hand2").pack(
            side="right", padx=4, ipadx=10, ipady=4,
        )
        tk.Button(btn_row, text="Save", command=do_reset, bg=self.GREEN, fg="black", cursor="hand2").pack(
            side="right", padx=4, ipadx=10, ipady=4,
        )

        top.bind("<Return>", lambda _e: do_reset())
        top.bind("<Escape>", lambda _e: top.destroy())
        pass_e.focus_set()

    def _delete_user(self) -> None:
        email = self._selected_email()
        if not email:
            return

        # Prevent locking yourself out
        sel_role = None
        for item in self.user_tree.get_children():
            vals = self.user_tree.item(item, "values")
            if vals and vals[0] == email:
                sel_role = vals[1]
                break

        if sel_role == "admin" and self._count_admins() <= 1:
            messagebox.showwarning(
                "Cannot Delete",
                "This is the last administrator account.\n"
                "Create another admin before deleting this one.",
            )
            return

        if not messagebox.askyesno(
            "Confirm Delete",
            f"Permanently delete user '{email}'?\n\nThis action cannot be undone.",
        ):
            return

        try:
            delete_user(email)
            self._refresh()
            messagebox.showinfo("Deleted", f"User '{email}' has been removed.")
        except Exception as exc:
            messagebox.showerror(
                "Delete Failed",
                f"Could not delete user '{email}':\n{exc}\n\n"
                "The user store may be locked or the record may not exist.",
            )

    def _toggle_admin(self) -> None:
        email = self._selected_email()
        if not email:
            return

        cur_role = None
        for item in self.user_tree.get_children():
            vals = self.user_tree.item(item, "values")
            if vals and vals[0] == email:
                cur_role = vals[1]
                break

        if cur_role == "admin" and self._count_admins() <= 1:
            messagebox.showwarning(
                "Cannot Demote",
                "There must be at least one administrator.\n"
                "Add another admin before demoting this account.",
            )
            return

        new_role = "user" if cur_role == "admin" else "admin"
        try:
            set_role(email, new_role)
            self._refresh()
            action = "promoted to admin" if new_role == "admin" else "demoted to user"
            messagebox.showinfo("Role Updated", f"'{email}' has been {action}.")
        except Exception as exc:
            messagebox.showerror(
                "Toggle Failed",
                f"Could not change role for '{email}':\n{exc}",
            )

    def _open_scoring(self) -> None:
        if callable(self.on_open_scoring):
            self.on_open_scoring()
