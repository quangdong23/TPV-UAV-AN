"""Canonical ten-object catalogue. Prompts are hypotheses, not verification rules."""
from enum import IntEnum

CATALOG_VERSION = "objects-v3-10"


class ClassId(IntEnum):
    PERSON = 1
    MOTORCYCLE = 2
    CAR = 3
    PLASTIC_TABLE = 4
    PLASTIC_CHAIR = 5
    VENDING_CART = 6
    ADVERTISING_SIGN = 7
    UTILITY_POLE = 8
    TREE_TRUNK = 9
    GOODS = 10


ADVANCED_OBJECT_CLASSES = [
    {"id": 1, "name_vi": "Người", "prompt": "person . seated person",
     "detail": "Người đứng, đi hoặc ngồi; không bắt buộc nhìn thấy toàn thân."},
    {"id": 2, "name_vi": "Xe máy", "prompt": "motorcycle . motor scooter",
     "detail": "Xe máy có động cơ, kể cả xe tay ga; hai bánh và tay lái không đủ phân biệt với xe đạp."},
    {"id": 3, "name_vi": "Xe ô tô", "prompt": "car . passenger car",
     "detail": "Ô tô con; bốn bánh và kính chắn gió cũng xuất hiện ở xe tải nên không dùng làm điều kiện phân biệt."},
    {"id": 4, "name_vi": "Bàn nhựa", "prompt": "plastic table . rectangular plastic table",
     "detail": "Bàn nhựa, kể cả mặt vuông/chữ nhật; không yêu cầu thấy đủ bốn chân. Chất liệu vẫn là dự đoán."},
    {"id": 5, "name_vi": "Ghế nhựa", "prompt": "plastic chair . low plastic stool",
     "detail": "Ghế nhựa có tựa và ghế đẩu thấp không tựa; không ràng buộc màu sắc."},
    {"id": 6, "name_vi": "Xe hàng rong / Xe đẩy bán hàng",
     "prompt": "street food cart . wheeled food vending cart . mobile food vending cart . fruit food stand",
     "detail": "Xe đẩy/quầy bán hàng di động; bao gồm sạp hàng hoa quả/thực phẩm vỉa hè."},
    {"id": 7, "name_vi": "Biển quảng cáo",
     "prompt": "shop sign . advertising signboard . signboard . hanging sign . advertising board . motel sign",
     "detail": "Biển hiệu/quảng cáo gắn mặt tiền, đặt trên đất hoặc treo trên cây/cột."},
    {"id": 8, "name_vi": "Cột điện", "prompt": "utility pole . electric pole with cables",
     "detail": "Cột điện/cáp; dây có thể bị che hoặc quá nhỏ, không bắt buộc thấy dây ở mọi box."},
    {"id": 9, "name_vi": "Thân cây", "prompt": "tree trunk . exposed tree trunk",
     "detail": "Phần thân cây nhìn thấy, kể cả thân nghiêng hoặc màu xám. Prompt không bảo đảm loại được box tán lá."},
    {"id": 10, "name_vi": "Hàng hóa",
     "prompt": "basket of fruit . crate of vegetables . fruit basket . stacked packaged products . merchandise display",
     "detail": "Nội dung hàng, sản phẩm đóng gói, sọt/thùng hoa quả và cụm trưng bày."},
]

# Reference combined caption; the CLI uses per-class or configured groups.
GROUNDING_DINO_PROMPT = " . ".join(
    phrase.strip() for item in ADVANCED_OBJECT_CLASSES
    for phrase in item["prompt"].split(".") if phrase.strip()
) + " ."
