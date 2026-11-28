import json, os, subprocess, sys, time
from pathlib import Path
APPROOT=Path("/home/claude/b2s"); APP=APPROOT/"app"; OUT=Path("/home/claude/fresh")
SRC=Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/1111_unfamiliar")
os.environ["PADDLE_PDX_CACHE_HOME"]=str(OUT/".paddlex"); os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"]="True"
photos=sorted(SRC.glob("IMG_*.HEIC")); step=sys.argv[1]
if step=="upload":
    import pillow_heif; pillow_heif.register_heif_opener()
    from PIL import Image, ImageOps
    for p in photos:
        d=OUT/"jobs"/p.stem; d.mkdir(parents=True,exist_ok=True)
        if (d/"preview.jpg").exists(): continue
        im=ImageOps.exif_transpose(Image.open(p)).convert("RGB"); im.thumbnail((4000,4000))
        im.save(d/"preview.jpg","JPEG",quality=92,optimize=True)
    print("upload done",len(photos))
elif step=="prepare":
    for p in photos:
        d=OUT/"jobs"/p.stem
        if (d/"preparation.json").exists(): continue
        t=time.time()
        r=subprocess.run([sys.executable,str(APP/"prepare_infer.py"),str(d/"preview.jpg"),str(d/"processed.jpg"),str(d/"preparation.json"),"--mode","auto"],cwd=APPROOT,capture_output=True,text=True,timeout=180)
        if r.returncode: print(p.stem,"FAILED",[l for l in r.stderr.splitlines() if "arn" not in l][-5:],flush=True); continue
        print(p.stem,f"{time.time()-t:.1f}s",json.loads((d/"preparation.json").read_text()).get("reason","")[:160],flush=True)
elif step=="ocr":
    model=sys.argv[2]; sys.path.insert(0,str(APP))
    from ocr_mapping import normalize
    from paddleocr import PaddleOCR
    rec={"baseline":APPROOT/"models/ocr/baseline_rec","tuned":APPROOT/"models/ocr/recognizer","downloaded":Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/_poster_eval_models/board2notes_reader_progression/00_downloaded_PP-OCRv5_server_rec"),"p40":Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/_poster_eval_models/board2notes_reader_progression/03_wide_input_still_40_columns_BROKEN"),"ft40":Path("/mnt/user-data/uploads/Computer Vision & Deep Learning/Assignment/_poster_eval_models/board2notes_reader_progression/01_finetuned_on_our_whiteboard_lines")}[model]
    pipe=PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det",text_detection_model_dir=str(APPROOT/"models/ocr/detector"),
        text_recognition_model_name="PP-OCRv5_server_rec",text_recognition_model_dir=str(rec),
        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
        text_det_limit_side_len=1600,text_det_limit_type="max",device="cpu",enable_mkldnn=False)
    for p in photos:
        d=OUT/"jobs"/p.stem
        conds=(("original",d/"preview.jpg"),("auto",d/"processed.jpg"),("auto_f35",d/"processed_f35.jpg"))
        if model in ("downloaded","ft40","p40"): conds=conds[:1]
        for cond,img in conds:
            o=d/f"ocr_{cond}_{model}.json"
            if o.exists() or not img.exists(): continue
            t=time.time(); res=list(pipe.predict(str(img)))[0].json["res"]
            try: n=normalize(res)
            except ValueError as e:
                if "No nonblank" not in str(e): raise
                n={"ocr":{"regions":[]},"region_evidence":[]}
            n["seconds"]=round(time.time()-t,1); o.write_text(json.dumps(n,ensure_ascii=False))
            print(p.stem,cond,model,len(n["ocr"]["regions"]),n["seconds"],flush=True)
if step=="prepare_f35":
    import pillow_heif; pillow_heif.register_heif_opener(); sys.path.insert(0,str(APP))
    from preprocessing.board_geometry import focal35_from_exif
    for p in photos:
        d=OUT/"jobs"/p.stem
        if (d/"preparation_f35.json").exists(): continue
        f=focal35_from_exif(p); t=time.time()
        r=subprocess.run([sys.executable,str(OUT/"prepare_f35.py"),str(d/"preview.jpg"),str(d/"processed_f35.jpg"),str(d/"preparation_f35.json"),"--mode","auto"],cwd=APP,capture_output=True,text=True,timeout=240,env=dict(os.environ,B2N_F35=str(f),PYTHONPATH=str(APP)))
        if r.returncode: print(p.stem,"FAILED",r.stderr[-300:],flush=True); continue
        lg=json.loads((d/"preparation_f35.json").read_text()); print(p.stem,f,f"{time.time()-t:.1f}s",[ (x["name"],x["applied"]) for x in lg.get("decisions",[])],flush=True)
if step=="detsize":
    model="tuned"; sys.path.insert(0,str(APP))
    from ocr_mapping import normalize
    from paddleocr import PaddleOCR
    side=int(sys.argv[2])
    pipe=PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det",text_detection_model_dir=str(APPROOT/"models/ocr/detector"),
        text_recognition_model_name="PP-OCRv5_server_rec",text_recognition_model_dir=str(APPROOT/"models/ocr/recognizer"),
        use_doc_orientation_classify=False,use_doc_unwarping=False,use_textline_orientation=False,
        text_det_limit_side_len=side,text_det_limit_type="max",device="cpu",enable_mkldnn=False)
    for p in photos:
        d=OUT/"jobs"/p.stem; o=d/f"ocr_det{side}_tuned.json"
        if o.exists(): continue
        t=time.time(); res=list(pipe.predict(str(d/"preview.jpg")))[0].json["res"]
        try: n=normalize(res)
        except ValueError: n={"ocr":{"regions":[]},"region_evidence":[]}
        n["seconds"]=round(time.time()-t,1); o.write_text(json.dumps(n,ensure_ascii=False)); print(p.stem,side,len(n["ocr"]["regions"]),n["seconds"],flush=True)
