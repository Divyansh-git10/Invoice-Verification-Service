"""Deterministic generator for synthetic Indian GST invoice fixtures.

Produces 15 invoices (PNG + PDF) with varied vendors, fonts, layouts and
total labels, plus edge cases (subtotal>total, comma/lakh grouping, paise
decimals, rotated / noisy / blurred scans, and one document with no
identifiable total). Writes a manifest.json with the ground-truth total
each invoice should yield (null where no total exists).

Run:  python3 tests/fixtures/generate_invoices.py
"""
import io
import json
import os
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "invoices")
MANIFEST = os.path.join(HERE, "manifest.json")

DPI = 200
W, H = 1654, 2339  # A4 at 200 DPI
MARGIN = 90
INK = (20, 20, 20)
MUTED = (90, 90, 90)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_FILES = {
    "sans": ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    "serif": ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    "mono": ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"),
    "cond": ("DejaVuSansCondensed.ttf", "DejaVuSansCondensed-Bold.ttf"),
}

Q = Decimal("0.01")


def font(key, size, bold=False):
    reg, bold_file = FONT_FILES[key]
    return ImageFont.truetype(os.path.join(FONT_DIR, bold_file if bold else reg), size)


def money(dec: Decimal, paise: bool = True) -> str:
    """Format a Decimal with Indian digit grouping (e.g. 1,00,000.00)."""
    dec = dec.quantize(Q, rounding=ROUND_HALF_UP)
    neg = dec < 0
    whole, frac = f"{abs(dec):.2f}".split(".")
    if len(whole) > 3:
        last3, rest, groups = whole[-3:], whole[:-3], []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        whole = ",".join(groups) + "," + last3
    out = whole + ("." + frac if paise else "")
    return ("-" if neg else "") + out


def canonical(dec: Decimal) -> str:
    return f"{dec.quantize(Q, rounding=ROUND_HALF_UP):.2f}"


def compute(items, gst_rate, discount=Decimal("0")):
    subtotal = sum(
        (Decimal(str(q)) * Decimal(str(r)) for _, _, q, r in items), Decimal("0")
    )
    taxable = subtotal - discount
    gst = (taxable * Decimal(gst_rate) / Decimal(100)).quantize(Q, ROUND_HALF_UP)
    cgst = (gst / 2).quantize(Q, ROUND_HALF_UP)
    sgst = gst - cgst
    total = taxable + cgst + sgst
    return {
        "subtotal": subtotal,
        "taxable": taxable,
        "discount": discount,
        "cgst": cgst,
        "sgst": sgst,
        "gst_rate": gst_rate,
        "total": total,
    }


def rtext(draw, right, y, text, fnt, fill=INK):
    w = draw.textlength(text, font=fnt)
    draw.text((right - w, y), text, font=fnt, fill=fill)


def ctext(draw, cx, y, text, fnt, fill=INK):
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


# ---------------------------------------------------------------------------
# Invoice rendering
# ---------------------------------------------------------------------------

def render_invoice(spec):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fk = spec["font"]
    cur = spec.get("cur", "Rs. ")
    v = spec["vendor"]
    t = compute(spec["items"], spec["gst"], Decimal(spec.get("discount", "0")))

    f_title = font(fk, 58, bold=True)
    f_vendor = font(fk, 46, bold=True)
    f_norm = font(fk, 30)
    f_small = font(fk, 26)
    f_bold = font(fk, 32, bold=True)
    f_total = font(fk, 42, bold=True)

    x0, x1 = MARGIN, W - MARGIN
    y = MARGIN

    modern = spec.get("template") == "modern"
    border = spec.get("template") != "compact"

    # Header
    if modern:
        d.rectangle([0, 0, W, 210], fill=(232, 238, 246))
        d.text((x0, 55), v["name"], font=f_vendor, fill=INK)
        d.text((x0, 120), v["address"], font=f_small, fill=MUTED)
        d.text((x0, 155), f"GSTIN: {v['gstin']}   State: {v['state']}", font=f_small, fill=MUTED)
        rtext(d, x1, 70, "TAX INVOICE", f_title)
        y = 250
    else:
        d.text((x0, y), v["name"], font=f_vendor, fill=INK)
        y += 60
        d.text((x0, y), v["address"], font=f_small, fill=MUTED)
        y += 38
        d.text((x0, y), f"GSTIN: {v['gstin']}   State: {v['state']}", font=f_small, fill=MUTED)
        y += 30
        ctext(d, W / 2, y + 20, "TAX INVOICE", f_title)
        y += 110

    d.line([x0, y, x1, y], fill=INK, width=2)
    y += 20

    # Meta row
    d.text((x0, y), f"Invoice No: {spec['invoice_no']}", font=f_norm, fill=INK)
    rtext(d, x1, y, f"Date: {spec['inv_date']}", f_norm)
    y += 44
    d.text((x0, y), f"Place of Supply: {v['state']}", font=f_small, fill=MUTED)
    y += 50

    # Items table
    cols = [
        ("#", 0.06, "l"),
        ("Description", 0.42, "l"),
        ("HSN", 0.12, "l"),
        ("Qty", 0.10, "r"),
        ("Rate", 0.14, "r"),
        ("Amount", 0.16, "r"),
    ]
    table_w = x1 - x0
    xs = []
    acc = x0
    for _, frac, _ in cols:
        xs.append(acc)
        acc += frac * table_w
    xs.append(x1)

    header_h = 50
    if border:
        d.rectangle([x0, y, x1, y + header_h], fill=(240, 240, 240), outline=INK)
    for i, (name, _, align) in enumerate(cols):
        cx0, cx1 = xs[i] + 10, xs[i + 1] - 10
        if align == "r":
            rtext(d, cx1, y + 10, name, f_bold)
        else:
            d.text((cx0, y + 10), name, font=f_bold, fill=INK)
    y += header_h

    row_h = 52
    for idx, (desc, hsn, qty, rate) in enumerate(spec["items"], start=1):
        amount = Decimal(str(qty)) * Decimal(str(rate))
        vals = [str(idx), desc, hsn, str(qty), money(Decimal(str(rate))), money(amount)]
        for i, (name, _, align) in enumerate(cols):
            cx0, cx1 = xs[i] + 10, xs[i + 1] - 10
            if align == "r":
                rtext(d, cx1, y + 12, vals[i], f_norm)
            else:
                d.text((cx0, y + 12), vals[i], font=f_norm, fill=INK)
        if border:
            d.line([x0, y + row_h, x1, y + row_h], fill=(200, 200, 200), width=1)
        y += row_h

    if border:
        d.rectangle([x0, y - row_h * len(spec["items"]) - header_h, x1, y], outline=INK)
        for xv in xs[1:-1]:
            d.line([xv, y - row_h * len(spec["items"]) - header_h, xv, y], fill=(200, 200, 200), width=1)

    y += 40

    # Totals block
    lines = [("Sub Total", money(t["subtotal"]))]
    if t["discount"] > 0:
        lines.append(("Discount", "-" + money(t["discount"])))
    lines.append((f"CGST @ {Decimal(spec['gst']) / 2:g}%", money(t["cgst"])))
    lines.append((f"SGST @ {Decimal(spec['gst']) / 2:g}%", money(t["sgst"])))

    # Wide block so long labels ("Total Amount", "Amount Payable") never
    # collide with the right-aligned currency amount.
    block_w = 900
    if modern:
        lx, rx = x0, x0 + block_w
    else:
        lx, rx = x1 - block_w, x1

    for label, val in lines:
        d.text((lx, y), label, font=f_norm, fill=MUTED)
        rtext(d, rx, y, cur + val, f_norm)
        y += 46

    y += 10
    d.line([lx, y, rx, y], fill=INK, width=2)
    y += 16
    box_top = y
    d.text((lx, y + 4), spec["label"], font=f_total, fill=INK)
    rtext(d, rx, y + 4, cur + money(t["total"]), f_total)
    y += 60
    d.rectangle([lx - 14, box_top - 8, rx + 14, y], outline=INK, width=3)

    # Footer
    d.text((x0, H - 150), "This is a computer-generated invoice.", font=f_small, fill=MUTED)
    d.text((x0, H - 110), "Subject to local jurisdiction.", font=f_small, fill=MUTED)

    return img, t


def render_delivery_note(spec):
    """A delivery challan with NO monetary total (no identifiable total)."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    fk = spec["font"]
    v = spec["vendor"]

    f_title = font(fk, 58, bold=True)
    f_vendor = font(fk, 46, bold=True)
    f_norm = font(fk, 32)
    f_small = font(fk, 26)
    f_bold = font(fk, 34, bold=True)

    x0, x1 = MARGIN, W - MARGIN
    y = MARGIN
    d.text((x0, y), v["name"], font=f_vendor, fill=INK)
    y += 60
    d.text((x0, y), v["address"], font=f_small, fill=MUTED)
    y += 38
    d.text((x0, y), f"GSTIN: {v['gstin']}   State: {v['state']}", font=f_small, fill=MUTED)
    y += 40
    ctext(d, W / 2, y, "DELIVERY CHALLAN", f_title)
    y += 110
    d.line([x0, y, x1, y], fill=INK, width=2)
    y += 24

    d.text((x0, y), f"Challan No: {spec['invoice_no']}", font=f_norm, fill=INK)
    rtext(d, x1, y, f"Date: {spec['inv_date']}", f_norm)
    y += 60

    d.text((x0, y), "Goods dispatched (for delivery only - not for sale):", font=f_small, fill=MUTED)
    y += 60

    d.rectangle([x0, y, x1, y + 50], fill=(240, 240, 240), outline=INK)
    d.text((x0 + 14, y + 10), "#", font=f_bold, fill=INK)
    d.text((x0 + 120, y + 10), "Description of Goods", font=f_bold, fill=INK)
    rtext(d, x1 - 20, y + 10, "Quantity", f_bold)
    y += 50

    for idx, (desc, qty) in enumerate(spec["items"], start=1):
        d.text((x0 + 14, y + 12), str(idx), font=f_norm, fill=INK)
        d.text((x0 + 120, y + 12), desc, font=f_norm, fill=INK)
        rtext(d, x1 - 20, y + 12, f"{qty} units", f_norm)
        d.line([x0, y + 56, x1, y + 56], fill=(200, 200, 200), width=1)
        y += 56

    y += 60
    d.text((x0, y), "Received the above goods in good condition.", font=f_norm, fill=INK)
    y += 120
    d.text((x0, y), "Receiver Signature: ______________________", font=f_small, fill=MUTED)
    return img, None


# ---------------------------------------------------------------------------
# Edge-case post-processing
# ---------------------------------------------------------------------------

def apply_edges(img, edges, rng):
    if "rotated" in edges:
        img = img.rotate(-7, expand=True, fillcolor="white", resample=Image.BICUBIC)
    if "noisy" in edges:
        arr = np.array(img).astype(np.int16)
        arr = arr + rng.normal(0, 24, arr.shape)
        # sprinkle salt & pepper
        mask = rng.random(arr.shape[:2])
        arr[mask < 0.01] = 0
        arr[mask > 0.99] = 255
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if "blurred" in edges:
        img = img.filter(ImageFilter.GaussianBlur(2.6))
    return img


# ---------------------------------------------------------------------------
# Vendor + invoice specifications
# ---------------------------------------------------------------------------

VENDORS = [
    {"name": "Bharat Steel Traders", "gstin": "27ABCDE1234F1Z5", "address": "14, MIDC Road, Andheri East, Mumbai 400093", "state": "Maharashtra"},
    {"name": "Ganesh Electronics", "gstin": "29PQRSX6789K2Z1", "address": "22, SP Road, Bengaluru 560002", "state": "Karnataka"},
    {"name": "Krishna Textiles Pvt Ltd", "gstin": "07LMNOP4567Q3Z8", "address": "5, Chandni Chowk, New Delhi 110006", "state": "Delhi"},
    {"name": "Surya Hardware & Tools", "gstin": "33UVWXY8901R4Z2", "address": "91, NSC Bose Road, Chennai 600079", "state": "Tamil Nadu"},
    {"name": "Deccan Auto Parts", "gstin": "36EFGHI2345S5Z9", "address": "7, General Bazaar, Secunderabad 500003", "state": "Telangana"},
    {"name": "Himalaya Pharma Distributors", "gstin": "06OPQRS0123U7Z4", "address": "3, Sector 18, Gurugram 122001", "state": "Haryana"},
    {"name": "Konark Machine Works", "gstin": "21JKLMN6789T6Z3", "address": "18, Industrial Estate, Bhubaneswar 751010", "state": "Odisha"},
    {"name": "Sagar Marine Supplies", "gstin": "24VWXYZ4567V8Z5", "address": "44, Ring Road, Surat 395002", "state": "Gujarat"},
    {"name": "Rajdhani Stationers", "gstin": "09ABCDE8901W9Z6", "address": "12, Aminabad, Lucknow 226018", "state": "Uttar Pradesh"},
    {"name": "Coastal Seafoods Ltd", "gstin": "32FGHIJ2345X1Z7", "address": "9, Harbour Road, Kochi 682001", "state": "Kerala"},
    {"name": "Malwa Agro Industries", "gstin": "23KLMNO6789Y2Z8", "address": "28, AB Road, Indore 452001", "state": "Madhya Pradesh"},
    {"name": "Nilgiri Logistics", "gstin": "29PQRST0123Z3Z9", "address": "6, Station Road, Mysuru 570001", "state": "Karnataka"},
    {"name": "Aravalli Cement Depot", "gstin": "08UVWXY4567A4Z1", "address": "51, MI Road, Jaipur 302001", "state": "Rajasthan"},
    {"name": "Vidarbha Power Systems", "gstin": "27ZABCD8901B5Z2", "address": "33, Wardha Road, Nagpur 440015", "state": "Maharashtra"},
    {"name": "Brahmaputra Tea Estate", "gstin": "18EFGHI2345C6Z3", "address": "2, GS Road, Guwahati 781005", "state": "Assam"},
]

SPECS = [
    dict(id="invoice_01", vendor=0, font="sans", cur="Rs. ", label="Grand Total", gst=18,
         template="classic", edges=[], fmt="png", invoice_no="BST/24-25/0142", inv_date="03 Apr 2025",
         items=[("MS Angle 50x50x6", "7216", 120, 132.50), ("Welding Rod 3.15mm", "8311", 40, 255)]),
    dict(id="invoice_02", vendor=1, font="serif", cur="Rs. ", label="Total Amount", gst=12,
         template="classic", edges=[], fmt="png", invoice_no="GE-2025-0876", inv_date="11 May 2025",
         items=[("LED Panel 40W", "9405", 60, 545), ("Copper Wire 2.5sqmm (roll)", "8544", 25, 1180)]),
    dict(id="invoice_03", vendor=2, font="mono", cur="Rs. ", label="Invoice Total", gst=5,
         template="compact", edges=[], fmt="pdf", invoice_no="KTX/0451", inv_date="19 Feb 2025",
         items=[("Cotton Fabric (m)", "5208", 300, 85), ("Polyester Thread (cone)", "5401", 50, 42)]),
    dict(id="invoice_04", vendor=3, font="cond", cur="Rs. ", label="Amount Payable", gst=18,
         template="classic", edges=[], fmt="png", invoice_no="SHT-9921", inv_date="27 Mar 2025",
         items=[("Hammer 1kg", "8205", 24, 410), ("Drill Bit Set", "8207", 15, 690),
                ("Measuring Tape 5m", "9017", 30, 150)]),
    dict(id="invoice_05", vendor=4, font="sans", cur="Rs. ", label="Grand Total", gst=18, discount="30000",
         template="classic", edges=["subtotal_gt_total"], fmt="png", invoice_no="DAP/25/0033", inv_date="08 Jan 2025",
         items=[("Industrial Pump 5HP", "8413", 4, 25000)]),
    dict(id="invoice_06", vendor=5, font="serif", cur="Rs. ", label="Total Amount", gst=18,
         template="modern", edges=["multiple_values"], fmt="png", invoice_no="HPD-2025-1180", inv_date="15 Jun 2025",
         items=[("Paracetamol 500mg (box)", "3004", 50, 120), ("Amoxicillin 250mg (strip)", "3004", 30, 240),
                ("Cough Syrup 100ml", "3004", 80, 65), ("Surgical Gloves (pack)", "4015", 40, 180),
                ("Face Mask (box)", "6307", 100, 95), ("Digital Thermometer", "9025", 25, 310),
                ("BP Monitor", "9018", 10, 1450), ("Hand Sanitizer 500ml", "3808", 60, 140),
                ("Vitamin C Tablets", "3004", 45, 85)]),
    dict(id="invoice_07", vendor=6, font="mono", cur="₹", label="Grand Total", gst=18,
         template="classic", edges=["comma_lakh"], fmt="png", invoice_no="KMW/0091", inv_date="02 Jul 2025",
         items=[("CNC Spindle Assembly", "8466", 3, 60000)]),
    dict(id="invoice_08", vendor=7, font="cond", cur="Rs. ", label="Amount Payable", gst=18,
         template="classic", edges=["decimals"], fmt="pdf", invoice_no="SMS-2025-0207", inv_date="21 May 2025",
         items=[("Nylon Rope 12mm (m)", "5607", 125, 37.50), ("Anchor Chain (m)", "7315", 30, 415.20)]),
    dict(id="invoice_09", vendor=4, font="sans", cur="Rs. ", label="Grand Total", gst=18,
         template="classic", edges=["rotated"], fmt="png", invoice_no="DAP/25/0061", inv_date="14 Apr 2025",
         items=[("Brake Pad Set", "8708", 40, 850), ("Oil Filter", "8421", 60, 220)]),
    dict(id="invoice_10", vendor=8, font="serif", cur="Rs. ", label="Invoice Total", gst=12,
         template="classic", edges=["noisy"], fmt="png", invoice_no="RS-2025-0455", inv_date="09 May 2025",
         items=[("A4 Paper Ream", "4802", 100, 285), ("Ball Pen (box)", "9608", 50, 180)]),
    dict(id="invoice_11", vendor=9, font="mono", cur="Rs. ", label="Total Amount", gst=5,
         template="classic", edges=["blurred"], fmt="png", invoice_no="CSF/2025/0188", inv_date="30 Jun 2025",
         items=[("Frozen Prawns (kg)", "0306", 200, 420), ("Fish Fillet (kg)", "0304", 150, 360)]),
    dict(id="invoice_12", vendor=11, font="cond", label="", gst=0,
         template="delivery", edges=["no_total"], fmt="png", invoice_no="DN-2026-0012", inv_date="05-08-2025",
         items=[("Wheat Seed Bags", 40), ("Fertilizer Sacks", 25), ("Pesticide Cans", 12),
                ("Irrigation Pipe Coils", 8)]),
    dict(id="invoice_13", vendor=12, font="sans", cur="₹", label="Grand Total", gst=18,
         template="modern", edges=["comma_lakh"], fmt="png", invoice_no="ACD-2025-0774", inv_date="17 Mar 2025",
         items=[("Cement Bag 50kg", "2523", 200, 395), ("Steel Rod 12mm (kg)", "7214", 150, 640)]),
    dict(id="invoice_14", vendor=13, font="serif", cur="₹", label="Amount Payable", gst=18,
         template="classic", edges=["comma_lakh"], fmt="png", invoice_no="VPS/2025/0044", inv_date="24 Feb 2025",
         items=[("Distribution Transformer 100kVA", "8504", 5, 180000)]),
    dict(id="invoice_15", vendor=14, font="mono", cur="Rs. ", label="Invoice Total", gst=18,
         template="classic", edges=["rotated", "noisy"], fmt="png", invoice_no="BTE-2025-0316", inv_date="12 Jul 2025",
         items=[("CTC Tea (kg)", "0902", 500, 240), ("Green Tea (kg)", "0902", 100, 480)]),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(42)
    manifest = {
        "generated_at": date.today().isoformat(),
        "currency": "INR",
        "locale": "en-IN",
        "count": len(SPECS),
        "note": "expected_total is the ground-truth invoice total a correct "
                "extractor should return (canonical decimal string). null means "
                "no identifiable total exists.",
        "invoices": [],
    }

    for spec in SPECS:
        spec = dict(spec)
        spec["vendor"] = VENDORS[spec["vendor"]]
        if spec["template"] == "delivery":
            img, totals = render_delivery_note(spec)
            expected = None
        else:
            img, totals = render_invoice(spec)
            expected = canonical(totals["total"])

        img = apply_edges(img, spec["edges"], rng)

        fmt = spec["fmt"]
        fname = f"{spec['id']}.{fmt}"
        path = os.path.join(OUT_DIR, fname)
        if fmt == "pdf":
            img.convert("RGB").save(path, "PDF", resolution=DPI)
            mime = "application/pdf"
        else:
            img.save(path, "PNG", dpi=(DPI, DPI))
            mime = "image/png"

        manifest["invoices"].append({
            "id": spec["id"],
            "file": f"invoices/{fname}",
            "mime_type": mime,
            "vendor": spec["vendor"]["name"],
            "font": spec["font"],
            "layout": spec["template"],
            "total_label": spec["label"] or None,
            "gst_rate_percent": spec["gst"],
            "expected_total": expected,
            "edge_cases": spec["edges"] or ["clean"],
        })

    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Wrote {len(SPECS)} invoices to {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
