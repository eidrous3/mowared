# موردين - Streamlit App

## المتطلبات
- Python 3.10+

## التثبيت
```bash
python -m venv .venv
source .venv/bin/activate  # على macOS/Linux
pip install -r requirements.txt
```

## التشغيل
```bash
streamlit run app.py
```

## الملفات
- data/purchases.csv: الفرع، المورد، التاريخ، القيمة
- data/payments.csv: الفرع، المورد، التاريخ، القيمة، مصدر الدفعة

## ملاحظات
- يمكن تعديل عدد الأشهر المعروضة في الواجهة الرئيسية من خلال تغيير `n_months=3` في `compute_dashboard`.
- اضغط على اسم المورد في جدول الهوم للدخول إلى تفاصيله.
