"""Normalize one PaddleOCR prediction JSON for the current slide planner.

Usage: python app/ocr_mapping.py INPUT.json OUTPUT.json
The model prompt should receive output["ocr"] only; the envelope retains evidence.
"""
import argparse
import json
from pathlib import Path

def normalize(record):
    texts=record.get("rec_texts")
    if not isinstance(texts,list):
        raise ValueError("Expected PaddleOCR rec_texts list")
    count=len(texts)
    for key in ("rec_scores","rec_polys","rec_boxes"):
        value=record.get(key)
        if value is not None and len(value)!=count:
            raise ValueError(f"{key} has {len(value)} entries; rec_texts has {count}")
    regions=[]
    details=[]
    for index,text in enumerate(texts):
        if not isinstance(text,str):
            raise ValueError(f"rec_texts[{index}] is not a string")
        if not text.strip():
            continue
        rid=f"r{len(regions)+1}"
        regions.append({"id":rid,"text":text.strip()})
        score=(record.get("rec_scores") or [None]*count)[index]
        poly=(record.get("rec_polys") or [None]*count)[index]
        box=(record.get("rec_boxes") or [None]*count)[index]
        details.append({"id":rid,"source_index":index,"confidence":score,
                        "polygon":poly,"box_xyxy":box})
    if not regions:
        raise ValueError("No nonblank recognized text")
    return {"source_type":"paddleocr_prediction","source_image":record.get("input_path"),
            "page_index":record.get("page_index"),
            "reading_order":"PaddleOCR rec_texts order; verify on multi-column boards",
            "ocr":{"regions":regions},"region_evidence":details}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    record=json.loads(args.input.read_text(encoding="utf-8"))
    result=normalize(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Saved {len(result['ocr']['regions'])} regions: {args.output}")

if __name__=="__main__":
    main()
