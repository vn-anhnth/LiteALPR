import io
import xml.etree.ElementTree as ET
import zipfile

z = zipfile.ZipFile(r"C:\Users\ADMIN\Downloads\FastPlateOCR_danh_muc_chinh_sua.docx")
root = ET.fromstring(z.read("word/document.xml"))
text = [node.text for node in root.iter() if node.text]
with io.open("extracted_docx.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(text))
