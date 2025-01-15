import easyocr

reader = easyocr.Reader(['en', 'ko'], gpu=False)
DATAPATH = "test_image/기부금영수증1.png"
result = reader.readtext(DATAPATH, detail = 0)
print(result)