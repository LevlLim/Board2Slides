import sys; sys.path.insert(0, '/home/claude/sweep')
from layers_common import *
sys.path.insert(0, '/home/claude/fresh/pkg/b2n_ocr'); os.environ['B2N_PACKAGE_ROOT'] = '/home/claude/fresh/pkg'
import ocr_backend as ob, cv2, glob
from pathlib import Path
imgs = sorted(glob.glob('/home/claude/diag/nw/b2n_data/newwriter/images/*.jpg'))[:4]
for mid in ('S0', 'G0', 'pretrained'):
    m, post = build(mid)
    ob.MODELS[mid] = Path(os.path.dirname(EXPORTS[mid])); rec = ob.RecognitionModel(mid, threads=2)
    for p in imgs:
        im = cv2.imread(p); rec.read([im])
        x = ob.resize_for_reader(im, rec.shape)
        rec.input.reshape([1, *x.shape]); rec.input.copy_from_cpu(x[None]); rec.predictor.run(); ref = rec.output.copy_to_cpu()[0]
        with paddle.no_grad(): out = m(paddle.to_tensor(x[None])).numpy()[0]
        print(mid, os.path.basename(p), 'max abs diff', float(np.abs(out - ref).max()), 'same argmax', float((out.argmax(-1) == ref.argmax(-1)).mean()))
