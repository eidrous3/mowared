import os
import urllib.parse
import pandas as pd
import streamlit as st
from datetime import datetime, date
from io import BytesIO
from typing import List, Tuple, Union, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PURCHASES_FILE = os.path.join(DATA_DIR, "purchases.csv")
PAYMENTS_FILE = os.path.join(DATA_DIR, "payments.csv")

PURCHASES_SCHEMA = ["branch", "vendor", "date", "amount"]
PAYMENTS_SCHEMA = ["branch", "vendor", "date", "amount", "source"]

CURRENCY_SUFFIX = " ج.م"
BRANCH_OPTIONS: List[str] = ["القاهرة", "الإسكندرية"]

@st.cache_data(show_spinner=False)
def load_csv(path: str, schema: List[str]) -> pd.DataFrame:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(path):
        df = pd.DataFrame(columns=schema)
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path)
    for col in schema:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df[schema]


def save_row(path: str, schema: List[str], row: dict) -> None:
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    file_exists = os.path.exists(path)
    df = pd.DataFrame([{k: row.get(k) for k in schema}])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    header = not file_exists
    df.to_csv(path, mode="a", header=header, index=False)
    load_csv.clear()


def get_last_n_months(reference_date: datetime, n: int = 3) -> List[Tuple[int, int, str]]:
    months = []
    year = reference_date.year
    month = reference_date.month
    for _ in range(n):
        months.append((year, month, f"{year}-{month:02d}"))
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    months.reverse()
    return months


def format_currency(val: Union[float, int, str]) -> str:
    try:
        num = float(val)
    except Exception:
        return ""
    formatted = f"{num:,.2f}"
    if formatted.endswith(".00"):
        formatted = formatted[:-3]
    return f"{formatted}{CURRENCY_SUFFIX}"


def compute_dashboard(purchases: pd.DataFrame, payments: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
    today = datetime.today()
    months = get_last_n_months(today, n_months)

    def ym_col(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df = df.copy()
            df["ym"] = pd.PeriodIndex([], freq="M")
            return df
        d = pd.to_datetime(df["date"], errors="coerce")
        df = df.copy()
        df["ym"] = d.dt.to_period("M").astype(str)
        return df

    p_pur = ym_col(purchases)
    p_pay = ym_col(payments)

    group_keys = ["branch", "vendor", "ym"]

    agg_pur = (
        p_pur.groupby(group_keys, as_index=False)["amount"].sum().rename(columns={"amount": "purchase"})
        if not p_pur.empty else pd.DataFrame(columns=[*group_keys, "purchase"])
    )
    agg_pay = (
        p_pay.groupby(group_keys, as_index=False)["amount"].sum().rename(columns={"amount": "payment"})
        if not p_pay.empty else pd.DataFrame(columns=[*group_keys, "payment"])
    )

    merged = pd.merge(agg_pur, agg_pay, on=group_keys, how="outer").fillna(0.0)

    base = merged[["branch", "vendor"]].drop_duplicates().reset_index(drop=True)

    for y, m, label in months:
        month_key = f"{y}-{m:02d}"
        sub = merged[merged["ym"] == month_key][["branch", "vendor", "purchase", "payment"]]
        base = base.merge(sub, on=["branch", "vendor"], how="left").fillna({"purchase": 0.0, "payment": 0.0})
        base.rename(columns={
            "purchase": f"{label} مشتريات",
            "payment": f"{label} دفعات",
        }, inplace=True)

    total_purchases = purchases.groupby(["branch", "vendor"], as_index=False)["amount"].sum().rename(columns={"amount": "total_purchases"}) if not purchases.empty else pd.DataFrame(columns=["branch", "vendor", "total_purchases"])
    total_payments = payments.groupby(["branch", "vendor"], as_index=False)["amount"].sum().rename(columns={"amount": "total_payments"}) if not payments.empty else pd.DataFrame(columns=["branch", "vendor", "total_payments"])

    base = base.merge(total_purchases, on=["branch", "vendor"], how="left").merge(total_payments, on=["branch", "vendor"], how="left").fillna({"total_purchases": 0.0, "total_payments": 0.0})
    base["debt"] = base["total_purchases"] - base["total_payments"]

    # Supplier share should be out of total positive debts only
    positive_debt = base["debt"].clip(lower=0.0)
    total_positive_debt = positive_debt.sum()
    if total_positive_debt != 0:
        base["نسبة المورد"] = (positive_debt / total_positive_debt * 100).round(2)
    else:
        base["نسبة المورد"] = 0.0

    # Display vendor name as plain text in the table (no markdown/URL text)
    base = base.copy()
    base["المورد"] = base["vendor"].astype(str)

    month_cols = []
    for _, _, label in months:
        month_cols.extend([f"{label} دفعات", f"{label} مشتريات"])  # payments then purchases

    cols = ["branch", "المورد", *month_cols, "debt", "نسبة المورد"]
    cols = [c for c in cols if c in base.columns]
    result = base[cols].copy()

    for c in result.columns:
        if c.endswith("مشتريات") or c.endswith("دفعات"):
            result[c] = result[c].apply(format_currency)

    # Add friendly debt column name and format
    if "debt" in result.columns:
        result.rename(columns={"debt": "إجمالي المديونية"}, inplace=True)
        result["إجمالي المديونية"] = result["إجمالي المديونية"].apply(format_currency)

    result.rename(columns={"branch": "الفرع"}, inplace=True)

    # Reorder for RTL visual preference: start from right with الفرع, end left with النسبة
    desired_order_ltr = [
        "نسبة المورد",
        "إجمالي المديونية",
        *[c for c in month_cols if c in result.columns],
        "المورد",
        "الفرع",
    ]
    desired_order_ltr = [c for c in desired_order_ltr if c in result.columns]
    result = result[desired_order_ltr]
    return result


def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def render_supplier_details(purchases: pd.DataFrame, payments: pd.DataFrame, supplier: Optional[str]):
    st.subheader("كشف حساب مورد")
    all_suppliers = sorted(set(purchases["vendor"].dropna().unique()).union(set(payments["vendor"].dropna().unique())))
    if not all_suppliers:
        st.info("لا توجد بيانات موردين بعد.")
        return
    default_supplier = supplier if supplier in all_suppliers else all_suppliers[0]
    supplier_name = st.selectbox(
        "اختر المورد",
        options=all_suppliers,
        index=(all_suppliers.index(default_supplier) if default_supplier in all_suppliers else 0),
    )

    branches = sorted(set(purchases["branch"].dropna().unique()).union(set(payments["branch"].dropna().unique())))
    branch_filter = st.multiselect("تصفية بالفروع", options=branches, default=branches)

    # Date range filter
    combined_dates = pd.to_datetime(pd.Series(list(purchases["date"].dropna().values) + list(payments["date"].dropna().values)), errors="coerce")
    if not combined_dates.empty and combined_dates.notna().any():
        min_d = combined_dates.min().date()
        max_d = combined_dates.max().date()
    else:
        today = date.today()
        min_d, max_d = date(today.year, today.month, 1), today
    date_range = st.date_input("نطاق التاريخ", value=(min_d, max_d))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_d, max_d

    p_sel = purchases[(purchases["vendor"] == supplier_name) & (purchases["branch"].isin(branch_filter))].copy()
    pay_sel = payments[(payments["vendor"] == supplier_name) & (payments["branch"].isin(branch_filter))].copy()

    p_sel = p_sel[(p_sel["date"] >= start_d) & (p_sel["date"] <= end_d)]
    pay_sel = pay_sel[(pay_sel["date"] >= start_d) & (pay_sel["date"] <= end_d)]

    p_sel["النوع"] = "مشتريات"
    pay_sel["النوع"] = "دفعات"

    p_sel.rename(columns={"branch": "الفرع", "vendor": "المورد", "date": "التاريخ", "amount": "القيمة"}, inplace=True)
    pay_sel.rename(columns={"branch": "الفرع", "vendor": "المورد", "date": "التاريخ", "amount": "القيمة", "source": "مصدر الدفعة"}, inplace=True)

    missing_source = set(p_sel.columns) ^ set(pay_sel.columns)
    for col in missing_source:
        if col not in p_sel.columns:
            p_sel[col] = ""
        if col not in pay_sel.columns:
            pay_sel[col] = ""

    details = pd.concat([p_sel, pay_sel], ignore_index=True)
    details.sort_values(by=["التاريخ", "النوع"], inplace=True)

    if "القيمة" in details.columns:
        details["القيمة"] = details["القيمة"].apply(format_currency)

    st.dataframe(details, use_container_width=True, hide_index=True)

    # Totals (use numeric originals for math)
    total_p = purchases[(purchases["vendor"] == supplier_name) & (purchases["branch"].isin(branch_filter)) & (purchases["date"] >= start_d) & (purchases["date"] <= end_d)]["amount"].sum()
    total_pay = payments[(payments["vendor"] == supplier_name) & (payments["branch"].isin(branch_filter)) & (payments["date"] >= start_d) & (payments["date"] <= end_d)]["amount"].sum()
    debt = total_p - total_pay

    st.markdown(f"**إجمالي المشتريات:** {format_currency(total_p)} | **إجمالي الدفعات:** {format_currency(total_pay)} | **المديونية:** {format_currency(debt)}")

    # Export button
    excel_btn = st.download_button(
        label="تصدير كشف حساب المورد",
        data=df_to_excel_bytes(pd.concat([p_sel.assign(_الرقم=range(1, len(p_sel)+1)), pay_sel.assign(_الرقم=range(1, len(pay_sel)+1))], ignore_index=True).drop(columns=[c for c in ["_الرقم"] if c in []]),
                               sheet_name=f"{supplier_name}"),
        file_name=f"supplier_{supplier_name}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def add_entry_forms():
    with st.sidebar:
        st.header("إضافة بيانات")
        tabs = st.tabs(["+ مشتريات", "+ دفعات"])

        with tabs[0]:
            branch = st.selectbox("الفرع", options=BRANCH_OPTIONS, index=0, key="p_branch")
            purchases_df = load_csv(PURCHASES_FILE, PURCHASES_SCHEMA)
            payments_df = load_csv(PAYMENTS_FILE, PAYMENTS_SCHEMA)
            all_vendors = sorted(set(purchases_df["vendor"].dropna().unique()).union(set(payments_df["vendor"].dropna().unique())))
            add_new_label = "إضافة مورد جديد..."
            vendor_options = [*all_vendors, add_new_label] if all_vendors else [add_new_label]
            vendor_choice = st.selectbox("المورد", options=vendor_options, index=0, key="p_vendor_choice")
            custom_vendor = ""
            if vendor_choice == add_new_label:
                custom_vendor = st.text_input("اسم المورد الجديد", key="p_vendor_new")
            date_val = st.date_input("التاريخ", value=datetime.today(), key="p_date")
            amount = st.number_input("القيمة", min_value=0.0, step=0.5, key="p_amount")
            if st.button("حفظ المشتريات", key="p_save"):
                final_vendor = (custom_vendor.strip() if vendor_choice == add_new_label else vendor_choice).strip()
                if not branch or not final_vendor:
                    st.warning("برجاء إدخال الفرع والمورد")
                else:
                    save_row(PURCHASES_FILE, PURCHASES_SCHEMA, {
                        "branch": branch.strip(),
                        "vendor": final_vendor,
                        "date": date_val,
                        "amount": amount,
                    })
                    st.success("تم حفظ المشتريات")

        with tabs[1]:
            branch = st.selectbox("الفرع", options=BRANCH_OPTIONS, index=0, key="pay_branch")
            purchases_df = load_csv(PURCHASES_FILE, PURCHASES_SCHEMA)
            payments_df = load_csv(PAYMENTS_FILE, PAYMENTS_SCHEMA)
            all_vendors = sorted(set(purchases_df["vendor"].dropna().unique()).union(set(payments_df["vendor"].dropna().unique())))
            add_new_label = "إضافة مورد جديد..."
            vendor_options = [*all_vendors, add_new_label] if all_vendors else [add_new_label]
            vendor_choice = st.selectbox("المورد", options=vendor_options, index=0, key="pay_vendor")
            custom_vendor = ""
            if vendor_choice == add_new_label:
                custom_vendor = st.text_input("اسم المورد الجديد", key="pay_vendor_new")
            date_val = st.date_input("التاريخ", value=datetime.today(), key="pay_date")
            amount = st.number_input("القيمة", min_value=0.0, step=0.5, key="pay_amount")
            source_options = [
                "حسام عبد الواحد",
                "حساب بنكي",
                "أحمد عبد الواحد",
                "محمد عطا",
            ]
            add_new_source_label = "إضافة مصدر جديد..."
            source_options_full = [*source_options, add_new_source_label]
            source_sel = st.selectbox("مصدر الدفعة", options=source_options_full, key="pay_source")
            custom_source = ""
            if source_sel == add_new_source_label:
                custom_source = st.text_input("اكتب مصدر الدفعة", key="pay_source_new")
            final_source = custom_source.strip() if source_sel == add_new_source_label else source_sel
            if st.button("حفظ الدفعة", key="pay_save"):
                final_vendor = (custom_vendor.strip() if vendor_choice == add_new_label else vendor_choice).strip()
                if not branch or not final_vendor:
                    st.warning("برجاء إدخال الفرع والمورد")
                elif not final_source:
                    st.warning("برجاء تحديد مصدر الدفعة")
                else:
                    save_row(PAYMENTS_FILE, PAYMENTS_SCHEMA, {
                        "branch": branch.strip(),
                        "vendor": final_vendor,
                        "date": date_val,
                        "amount": amount,
                        "source": final_source,
                    })
                    st.success("تم حفظ الدفعة")


def main():
    st.set_page_config(page_title="موردين", layout="wide")
    # RTL + Arabic font (Noto Sans Arabic)
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
        :root, html, body, .stApp, [class^="css"], [class*="css"], * {
            font-family: 'Noto Sans Arabic', sans-serif !important;
        }
        .stApp { direction: rtl; text-align: right; }
        .stMarkdown, .stText, .stSelectbox, .stNumberInput, .stDateInput, .stButton, label, .stDataFrame { direction: rtl; text-align: right; }
        thead tr th { text-align: right !important; }
        /* Keep download buttons on one line and widen */
        .stDownloadButton > button { white-space: nowrap; min-width: 140px; margin: 0 12px; }
        /* Ensure primary buttons like refresh have enough width and spacing */
        .stButton > button { min-width: 120px; white-space: nowrap; margin: 0 12px; }
        /* Smaller KPI numbers and labels */
        div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        div[data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
        /* KPI card styling */
        div[data-testid="stMetric"] { background: #f7f7fb; border: 1px solid rgba(0,0,0,0.08); padding: 10px 12px; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Simple passkey gate
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if not st.session_state.auth_ok:
        st.subheader("الرجاء إدخال رمز الدخول")
        key_input = st.text_input("رمز الدخول", type="password")
        if st.button("دخول"):
            if key_input == "1319":
                st.session_state.auth_ok = True
                st.success("تم التحقق")
                st.rerun()
            else:
                st.error("رمز غير صحيح")
        st.stop()
    # Title removed as requested

    purchases = load_csv(PURCHASES_FILE, PURCHASES_SCHEMA)
    payments = load_csv(PAYMENTS_FILE, PAYMENTS_SCHEMA)

    add_entry_forms()

    st.subheader("نظرة عامة")

    # KPI cards directly under the overview title
    total_p_all = purchases["amount"].sum() if not purchases.empty else 0.0
    total_pay_all = payments["amount"].sum() if not payments.empty else 0.0
    debt_all = total_p_all - total_pay_all

    p_by_branch = (purchases.groupby("branch")["amount"].sum() if not purchases.empty else pd.Series(dtype=float))
    pay_by_branch = (payments.groupby("branch")["amount"].sum() if not payments.empty else pd.Series(dtype=float))
    branches = sorted(set(list(p_by_branch.index) + list(pay_by_branch.index)))
    debt_by_branch = {b: float(p_by_branch.get(b, 0.0) - pay_by_branch.get(b, 0.0)) for b in branches}

    today = datetime.today()
    this_month = pd.Period(today.strftime("%Y-%m"), freq="M")
    if not purchases.empty:
        p_month = purchases.copy()
        p_month["_ym"] = pd.to_datetime(p_month["date"], errors="coerce").dt.to_period("M")
        p_month = p_month[p_month["_ym"] == str(this_month)]
        p_month_by_branch = p_month.groupby("branch")["amount"].sum() if not p_month.empty else pd.Series(dtype=float)
    else:
        p_month_by_branch = pd.Series(dtype=float)

    kpi_cols_top = st.columns(5)
    with kpi_cols_top[0]:
        st.metric(label="إجمالي المديونية", value=format_currency(debt_all))
    with kpi_cols_top[1]:
        st.metric(label="مديونية الإسكندرية", value=format_currency(debt_by_branch.get("الإسكندرية", 0.0)))
    with kpi_cols_top[2]:
        st.metric(label="مديونية القاهرة", value=format_currency(debt_by_branch.get("القاهرة", 0.0)))
    with kpi_cols_top[3]:
        st.metric(label="مشتريات الإسكندرية (الشهر)", value=format_currency(float(p_month_by_branch.get("الإسكندرية", 0.0))))
    with kpi_cols_top[4]:
        st.metric(label="مشتريات القاهرة (الشهر)", value=format_currency(float(p_month_by_branch.get("القاهرة", 0.0))))
    ctrl = st.container()
    with ctrl:
        # Single row: [Export button (far left)] [spacer] [Months dropdown (far right, no label)]
        col_btn, spacer, col_months = st.columns([4, 4, 4])
        with col_months:
            n_months = st.selectbox("", options=list(range(1, 13)), index=2, label_visibility="collapsed")
        # Compute after selection
        dashboard_df = compute_dashboard(purchases, payments, n_months=n_months)
        with col_btn:
            btn_cols = st.columns([1, 1])
            with btn_cols[0]:
                st.download_button(
                    label="تصدير إكسيل",
                    data=df_to_excel_bytes(dashboard_df, sheet_name="Dashboard"),
                    file_name="dashboard.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with btn_cols[1]:
                if st.button("تحديث", type="primary", key="refresh_inline"):
                    load_csv.clear()
                    st.rerun()

    st.dataframe(dashboard_df, use_container_width=True, hide_index=True)

    params = st.query_params
    supplier_param = params.get("supplier")
    if isinstance(supplier_param, list):
        supplier_param = supplier_param[0] if supplier_param else None

    st.divider()
    render_supplier_details(purchases, payments, supplier_param)


if __name__ == "__main__":
    main()
