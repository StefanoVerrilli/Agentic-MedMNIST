"""Patch a textutil-generated .docx to embed architecture.png as a proper
OOXML inline image, replacing the DIAGRAMPLACEHOLDER marker."""
import re, shutil, zipfile
from pathlib import Path

SRC, OUT, IMG = "base.docx", Path("../../HAT-MedMNIST_Project_Proposal.docx"), "architecture.png"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"

# image size: 1400x652 px -> 6.0in wide, aspect-preserved
CX = 5486400
CY = round(CX * 652 / 1400)

DRAWING = f'''<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">\
<wp:extent cx="{CX}" cy="{CY}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>\
<wp:docPr id="1" name="Architecture"/>\
<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>\
<a:graphic xmlns:a="{A}"><a:graphicData uri="{PIC}">\
<pic:pic xmlns:pic="{PIC}"><pic:nvPicPr><pic:cNvPr id="1" name="architecture.png"/>\
<pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdImg1"/>\
<a:stretch><a:fillRect/></a:stretch></pic:blipFill>\
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{CX}" cy="{CY}"/></a:xfrm>\
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>\
</a:graphicData></a:graphic></wp:inline></w:drawing>'''

zin = zipfile.ZipFile(SRC)
names = zin.namelist()
data = {n: zin.read(n) for n in names}
zin.close()

# 1) document.xml: add namespaces + swap marker run text for the drawing
doc = data["word/document.xml"].decode("utf-8")
for ns, uri in (("a", A), ("pic", PIC), ("wp", WP)):
    if f'xmlns:{ns}=' not in doc[:800]:
        doc = doc.replace("<w:document ", f'<w:document xmlns:{ns}="{uri}" ', 1)
doc, n = re.subn(r"<w:t[^>]*>DIAGRAMPLACEHOLDER</w:t>", DRAWING, doc)
assert n == 1, f"marker replacements: {n}"
data["word/document.xml"] = doc.encode("utf-8")

# 2) relationships: add image rel
rels = data["word/_rels/document.xml.rels"].decode("utf-8")
rel = ('<Relationship Id="rIdImg1" Type="http://schemas.openxmlformats.org/'
       'officeDocument/2006/relationships/image" Target="media/image1.png"/>')
data["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", rel + "</Relationships>").encode()

# 3) content types: ensure png default
ct = data["[Content_Types].xml"].decode("utf-8")
if 'Extension="png"' not in ct:
    ct = ct.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
data["[Content_Types].xml"] = ct.encode()

# 4) add the media part
data["word/media/image1.png"] = Path(IMG).read_bytes()

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in data.items():
        z.writestr(n, b)
print("patched ->", OUT)
