# Board2Notes — master project record

Last updated: **8 November 2026, Singapore time**  
Status: **Working draft for the team to extend.** Compiled from saved results, independent reviews and the project discussion. This is a project memory, not the final submission.

**How to revise from this document:** read the explanatory paragraphs in sections 1, 3, 4, 5 and 6 in order. They tell the story of why we changed direction. Use the tables when you need exact numbers, and section 12 to practise explaining the work aloud. You do not need to memorise every model name or score.

## Read this first

We are building a system that turns photographs of handwritten whiteboard notes into structured notes and presentation slides.

The biggest recent findings are:

1. Our whiteboard data and adapted recognition recipe substantially improve whiteboard reading compared with the untouched pretrained reader.
2. Matching the recogniser's training and inference sequence geometry matters. Our controlled P40/P128 experiment showed a very large difference at the tested wide input size.
3. Layer-wise learning rates (S1) are promising, but no S0–S3 strategy passed the agreed primary validation rule.
4. Allowing CTC loss to update the backbone (G0) improved the two evaluated unfamiliar writers in both seeds. It did not establish an overall validation improvement and worsened IAM paper handwriting.
5. Super-resolution did not reliably reduce OCR error in the configurations tested. It is paused.
6. We have preprocessing code, but automatic perspective selection and full-photo benefit are not established sufficiently to call every component production-ready.
7. **8 Nov update:** the six-reader evaluation on the fresh Jy originals is complete. S0 averages 7.34% CER, S1 5.52% and G0 5.64%. G0 passes the confirmation rule for this collection; S1 also improves in the secondary comparison. The original validation verdict remains unchanged.

**Current position:** retain S0 as the clean comparison baseline; S1 and G0 now have supporting evidence from the Jy collection. A local OCR-only prototype is working, and 16 full-photo prediction sets are available. Finish full-photo ground-truth completeness/scope review and choose a fixed reader/preprocessing policy before final integration. There is no final, universally superior model established by these experiments.

### How to maintain this file

- Add a dated entry after every meaningful experiment or integration change.
- Keep unsuccessful results: they explain our decisions.
- Link to the exact result folder, config and notebook; do not rely on chat recollections.
- Distinguish **verified result**, **historical result**, **team-reported status**, and **proposed work**.
- Correct an earlier claim explicitly rather than silently rewriting the history.
- When Claude or another teammate adds information, include its evidence and author/date. Predictions in a planning message are not completed findings.

## 1. Goal, professor feedback and rubric

### Goal

Intended app flow:

```text
Whiteboard photo
    -> optional, justified image preprocessing
    -> text detection and line cropping
    -> handwriting recognition
    -> OCR text with positions/confidence
    -> LLM structures the notes
    -> slide converter creates a presentation
```

The exact deployed interfaces and model choices still need to be recorded by the team.

### Professor feedback — 5 November 2026

As recorded in our discussion, the professor described our preprocessing contribution as “super limited” and asked us to investigate PaddleOCR itself and adapt its models/settings to the whiteboard-handwriting task.

Our response was to investigate the recogniser's sequence representation and training strategy, rather than only add image corrections or train for longer.

To understand the change in direction, imagine what we would show the professor before this feedback. We could demonstrate that we cropped a photo, corrected its angle or made it brighter. Those are useful image-processing operations, but they do not explain how we adapted the neural network to our particular handwriting. The feedback pushed us to ask a deeper question: **what is happening inside PaddleOCR, and which part of its existing design or training recipe needs adapting for our task?**

That led to two kinds of investigation. First, we examined how the model compresses a text line into a sequence of visual features: this became the pooling experiment. Next, we examined how the model learns from its mistakes: this became the learning-rate, focal-loss and guidance experiments. Each experiment needs a reason and a comparison, even if its result is inconclusive.

### Rubric implications

The briefing covers Dataset, Model Development, Deployment, UI and Scalability. Accuracy is not a separate rubric row, but measured results are evidence that our choices are justified.

| Area | Our work/evidence | Still to confirm |
|---|---|---|
| Dataset | Own photos, transcriptions, crop review, augmentation, split and leakage checks | Final writer IDs, label-review coverage and submission inventory |
| Model development | Pooling adaptation; matched two-seed LR, focal-loss and gradient-routing studies | Final presentation of the hypotheses, limitations and chosen model |
| Deployment | Exported recognition models with checkpoint-equivalence checks | Working local and cloud deployment, optimisation and measurements |
| UI | Team app work | Current implementation and a complete user journey |
| Scalability | Reusable preprocessing/training code and caching are available | Actual runtime, memory, concurrency and deployment evidence |

The Excellent model-development wording refers to architectures not previously used for similar tasks or novel training strategies. Changing a built-in setting does not automatically establish novelty or guarantee a grade. Explain the task-specific diagnosis, implementation and controlled evidence accurately.

Sources: [ProjectBriefing.pptx](../ProjectBriefing.pptx), [CVDL Hurdle 1.pdf](CVDL%20Hurdle%201.pdf). Add the team's final interpretation of the lecturer's expectations if it changes.

## 2. Workstreams and ownership

This table records the division discussed so far; the team should confirm names and current status.

| Workstream | Ownership discussed | Current record |
|---|---|---|
| Dataset collection and annotation | Everyone contributes; ZB coordinates related preparation | Own whiteboard crops and sentence data available |
| Preprocessing and earlier SR investigation | ZB, with lighting/rotation helpers from Jian Yan | Implemented and experimentally investigated; deployment policy needs confirmation |
| Recognition model development | Teammate fine-tuning work plus the reviewed V/S/G experiments | Completed runs and exported models available |
| VLM comparison | Levl, as previously discussed | Team to add exact model, dataset and results |
| LLM structuring | Teammate | Team-reported work; add exact input/output contract and evidence |
| Presentation conversion | Teammate, using pptxGenJS | Team reported a demo accepting text/JSON; add the current local folder and integration status |
| App, local/cloud deployment and scalability | Confirm owners | Add demonstrable status; do not infer these are complete or absent |

## 3. Dataset: what we collected and what the labels mean

Our first challenge was to provide the right kind of examples. A full whiteboard photograph contains writing, background, reflections and sometimes objects. A detector learns **where the writing is**; a recogniser learns **what the writing says**. These are different learning problems, so the annotation must match the task.

For example, a polygon around a line tells the detector where that line is. To train the recogniser, we crop the line and pair it with a label such as `The operating system manages memory`. That label must describe the pixels actually inside the crop. If the first word is cut off or a second line is included, merely typing the intended sentence does not repair the training example. We must fix the crop or exclude it from that particular use.

This explains why so much of our work involved reviewing crops and labels. A recogniser learns from the examples we provide. If we give it an incomplete image with a complete-sentence label, we ask it to predict characters that are not visible. Better annotation can therefore matter more than another training setting.

### Annotation types

| Training task | Required annotation |
|---|---|
| Text detection | Text-region polygons/boxes on full photographs |
| Text recognition | A cropped word/line and its exact transcription |
| Paired image restoration/SR | Corresponding low-quality input and clear target, with suitable geometric alignment |
| Final OCR evaluation | Reviewed ground truth; full-photo evaluation must also account for missed/merged lines |

A box labelled only `text` is useful for detection but does not provide the words needed to train a recogniser. A Parquet file is a data container: its image bytes can store the crop and its text field can store the transcription. Image bytes do not inherently mean bounding-box coordinates.

### Sources used or investigated

- **IAM:** existing paper-handwriting recognition data. Useful reference, but handwriting/capture conditions differ from our whiteboards. Included in some earlier training runs and retained as an out-of-domain evaluation.
- **Own whiteboard photos:** lecture/CNN board, Airdrop and 17 Oct collections; close, medium, distant and side-angle views. Repeated photos are not independent handwriting samples.
- **Long sentences:** collected to address the shortage of longer training labels and evaluate unseen sentences in familiar handwriting.
- **Teammates' 64 crops:** used in older V-runs. Writer identities are uncertain, so they were excluded from the newer S/G training batches.
- **New-writer evaluation:** JH folders and `Dataset(train,test,label,original photo)` represent two different writers according to the user. Neither wrote the lecture/CNN photos. Current evaluation: 160 crops, two writers.
- **HierText/other public sources:** investigated as additional data. Do not describe them as included in the completed S/G training; those runs used our clean own-data list.
- **Jy Dataset, inspected/evaluated 8 Nov:** user identified it as unseen and subsequently confirmed teammate label review. It contains 16 full photos, 298 original crops and seven variants per original (2,384 labelled files). It remains excluded from training. Its supplied train/test folders share 254 original families, so they are not a clean training/evaluation split. All 66 supplied test originals have a transformed sibling in train. Original-file matching found no exact byte overlap with the checked prior packages. The initial [Jy inventory review](codex_jy_dataset_review_20261108/REVIEW.md) preceded inference; the later six-reader result is recorded in section 6D below. Writer provenance remains user supplied; crop-label review and complete-photo annotation are different checks.

### Current counts versus historical inventories

| Item | Count | Meaning |
|---|---:|---|
| Stage-1 reader inventory, 5 Nov | 6,090 images | Includes original and augmented entries; not 6,090 independent handwritten lines |
| Current S/G training list | 1,359 entries | Same examples across strategies, after preparation/label-length filters |
| Whiteboard validation | 136 crops, 26 photo clusters | Chooses checkpoints and is the primary strategy-comparison set |
| CNN-board evaluation | 174 crops, 38 photo clusters, 7 texts | Many crops repeat the same small set of phrases |
| Held-out sentence evaluation | 57 crops, 7 sentences | Same writer/board as the sentence training collection |
| Long-only reporting view | 41 of those 57 crops, 5 sentences | A subset, not an independent additional test |
| New-writer evaluation | 160 crops, 40 photo clusters | DS: 70 crops; JH: 90 crops |
| IAM evaluation | 976 lines | Out-of-domain handwriting check |

Earlier inventories show 1,146/139/179 whiteboard train/validation/test crops and 281 sentence crops. Use the prepared manifests for the exact effective counts of each run; do not mix source inventory counts with evaluated counts.

### Split and review rules

Training, validation and testing play different roles. **Training** is where the model sees labelled examples and updates its parameters. **Validation** is our practice assessment: it helps choose the checkpoint and compare planned settings. **Testing** asks whether the chosen method works on examples kept aside for evaluation. If we repeatedly inspect test mistakes and use them to design the next model, that set starts functioning as development data.

“Held-out sentence” simply means that we kept that sentence and its related photos out of training. It does not necessarily mean unfamiliar handwriting. In our sentence holdout, the person and board are familiar but the sentences are new. The new-writer set asks a harder, different question: can the model read handwriting from someone whose examples were excluded from training?

Taking ten photographs of the same sentence gives ten useful capture examples, but it does not give ten different sentences or ten writers. Augmentation also adds variations of existing examples rather than new handwriting identities. This distinction explains why we can have many crop files and still need more writer variety.

1. Augmented copies inherit their original crop's split. No validation/test descendants enter training.
2. Hold out all photos and augmented copies of the selected test sentences.
3. Exclude the 64 uncertain-writer teammates' originals and descendants from clean S/G training. Start from the original pretrained weights, not a checkpoint that previously saw those crops.
4. Check exact image/text overlap. Writer identity additionally requires provenance; image hashes cannot establish it.
5. Review labels against the pixels. OCR drafts are suggestions, not ground truth.
6. Watch for clipped characters, extra lines, empty margins, incorrect pairing and blurred or misaligned clear targets.
7. Do not select test crops simply because the baseline already reads them correctly. Some older experiments had that selection bias; their conclusions apply only to those selected crops.

**Label-review update — 8 Nov, added by Codex after checking Claude's supplied evidence:** [REVIEW_STATUS.txt](board2notes_newwriter_test_20261107/REVIEW_STATUS.txt) records that at 15:55 on 7 Nov, ZB reviewed the image/label page and reported all 160 new-writer labels and all 57 held-out sentence labels correct. The new-writer reviewed label file is byte-identical to the original labels, and the 57 reviewed sentence labels match the evaluated filenames/transcriptions. The review page contains images/labels without prediction fields. This verifies the saved review record and label identity; it is not a second independent human transcription audit. The folder's older README statement “NOT yet reviewed” is superseded by this dated status record.

That resolves the earlier open item for those **evaluation labels**. It does not establish that every other training-sentence label received the same personal review. Record the training-label review coverage separately.

### How the new-writer evaluation was assembled

The [builder README](board2notes_newwriter_test_20261107/README.md) and [manifest](board2notes_newwriter_test_20261107/manifest.csv) document 59 original sentence crops and 31 original short-word crops from JH, plus 70 original DS crops. Pre-made augmented variants were excluded. The original source-folder `train`/`test` names are preserved as provenance; **all 160 are evaluation examples in our S/G study**, regardless of those folder names.

The README records four skipped short-word families without an original: JH9.1, JH17.1, JH24.1 and JH25.1. Checks against the pinned dictionary and packaged training labels confirm that all new-writer label characters are supported and no normalised complete label text overlaps the whiteboard, long-sentence, teammates' or IAM training-label sets. This is a complete-label check, not proof that every individual word is new.

Five manifest entries retain image-quality/independence warnings: sentence crops JH1.2, JH5.3 and JH7.3 contain other writing near an edge; short-word crops JH19.1 and JH22.1 may show the same capture. Correct transcription does not remove these crop or dependence concerns. Keep the original full-set result and consider a clearly labelled sensitivity analysis if needed; do not silently filter after seeing scores.

Evidence: [stage-1 inventory](board2notes_reader_dataset_20261105/README_stage1.md), the immutable data ZIPs and the `data/prepared.json` files inside each result package. Some stage-1 “text-area fit” estimates were later treated as diagnostics, not hard CTC impossibility rules: distinguish total output positions from positions covering visible text.

## 4. Earlier preprocessing, restoration and fusion work

These experiments are historical context. Their crops, readers and scoring differ from the current S/G experiments, so their CER values must not be put into one common ranking.

We initially tried to help OCR by improving the image it receives. This was a reasonable starting point: side-angle photographs distort the writing, distant writing becomes small, and illumination can hide faint strokes. However, every correction can also damage an image. Perspective correction stretches and resamples pixels; rotation can follow the wrong line; aggressive cleaning can remove genuine handwriting. That is why we moved toward deciding whether a correction is needed for each photograph and measuring the OCR result afterwards.

Perspective correction uses geometric evidence about the board. Parallel physical board edges can converge toward a vanishing point in a photograph. With suitable camera information and a vertical reference or assumption, that helps estimate the board's orientation. A homography then maps the original pixel coordinates into a corrected view. The vanishing point provides evidence; the homography performs the mapping. If the selected edges are actually handwriting strokes or room features, the calculation can be mathematically valid but correct the wrong geometry.

Our “zoom-in” work involved several separate steps. We could detect writing on a smaller preview, map those coordinates back to the original photograph, and crop the original pixels. **Cropping selects a region; enlargement happens only when we subsequently resample that crop to more pixels.** This distinction matters because the cropped region may already contain enough detail without an extra upscaler.

Super-resolution went further: a neural network tried to predict a clearer, higher-resolution image. We learned that an image can look sharper without becoming easier to read. A restored stroke might look convincing but change an `n` into an `m`, for example. Pixel similarity scores such as PSNR therefore could not decide success; we needed to check whether PaddleOCR made fewer character mistakes. Our tested SR versions did not establish a reliable OCR benefit, so we paused that direction rather than keep making the images look better without improving the reading.

Multi-photo selection explored another possibility: read the same line from several photographs and keep a suitable reading. It showed a promising historical result, but the test already knew which crops showed the same line. An app must establish those correspondences automatically, so a successful crop-level test does not mean that the complete feature is finished.

| Approach | What we did | What the evidence supports | Current decision |
|---|---|---|---|
| Perspective correction | Board-region estimation, physical edge detection, vanishing-point/camera geometry, homography warp | Can help some side views; wrong geometry or resampling can harm other photos | Keep investigating conservative geometry checks; no blanket correction |
| Lighting | Investigated dimness, contrast and uneven illumination; gentle corrections | A cleaner appearance alone does not prove better OCR | Conditional candidate; measure with the chosen reader |
| Rotation | Investigated text-row agreement and glyph directions instead of following room lines | Code has conservative abstention and expanded-canvas safeguards | Full OCR benefit still needs evidence for the active rule |
| Crop enlargement | Detect region, map coordinates to the original-resolution photo, crop, optionally enlarge | Retains more useful text pixels for the reader than shrinking the whole scene | Keep separate from learned SR and test against a no-enlargement arm |
| Blemish removal | Residue filtering/binarisation experiments | Earlier comparisons harmed OCR | Excluded from the current first-stage preprocessing comparison |
| Learned SR | Custom CNN restorers, stroke/readability losses, pretrained SR and text-prior experiments | No reliable OCR benefit established in the tested settings; better PSNR/sharpness did not guarantee lower CER | Paused |
| Multi-photo reading selection | Compare readings of the same line across photos, select highest-confidence reading | Historical B3 result: 12.8% -> 8.0% mean CER with three-photo selection | Promising separate feature; automatic line matching and app capture flow remain requirements |

Crop enlargement is a preprocessing operation. Randomly creating altered training examples is augmentation. A learned SR model predicts additional pixels; bicubic interpolation simply resamples existing pixels. Keep these terms separate.

The fusion test used known line correspondences from the labels, PP-OCRv6 and a small seven-text board. It does not establish automatic full-photo fusion, and confidence can still select wrong readings.

Main evidence: [SR and fusion record](whiteboard-ocr/docs/SUPER_RESOLUTION_RUNS.md), [preprocessing package](whiteboard-ocr/preprocessing/README.md).

### Current preprocessing notebooks

- [07](whiteboard-ocr/notebooks/07_preprocessing_pipeline.ipynb): dataset/pipeline work.
- [08](whiteboard-ocr/notebooks/08_board2notes_preprocessing_pipeline.ipynb): independent preprocessing comparison, decision signals and visual examples.
- [09 manual geometry](whiteboard-ocr/notebooks/09_manual_geometry_diagnostic.ipynb): manual versus automatic geometry diagnostic on eight development photos; only two have sufficient observed geometry for manual rectification.

Notebook 08's round-4 arms use the names S0–S4 for **preprocessing variants**. Those names are separate from the **training strategies S0–S3** below. Always include the notebook/experiment name when discussing an arm.

**To confirm before shipping:** latest round-4 results, the chosen preprocessing chain, active decision thresholds and performance with the final exported reader. Some geometry diagnostics use a pretrained PP-OCRv6 reader, whereas the newer training experiments concern PP-OCRv5_server_rec.

## 5. The recogniser: the concepts behind our experiments

PaddleOCR's recogniser is one connected neural network with several named parts. **Backbone, neck and head describe their jobs.** The backbone extracts visual features; the neck processes those features into a useful representation; the head produces the prediction. These names organise familiar CNN and attention operations rather than introduce an entirely different kind of learning.

During training the two recognition branches share backbone features:

```text
Line crop -> HGNetV2 backbone -> visual features
                                  |-> SVTR neck -> CTC head -> CTC loss
                                  |-> NRTR branch           -> NRTR loss
```

- **Backbone:** CNN extracting strokes and shapes.
- **SVTR neck:** processes relationships between visual positions with convolution/attention. It is already trained in every S/G run.
- **CTC head:** predicts character probabilities and blanks along a horizontal sequence. This is the branch used for inference.
- **NRTR branch:** attention decoder using image features and text prefixes during training; supplies auxiliary supervision. It is not the deployed text postprocessor.
- **Loss:** measures prediction error using probabilities and guides parameter updates. Correct predictions can still have nonzero loss when probability is below one.
- **Optimiser:** applies the resulting updates. Our S/G runs use Adam.

For a concrete example, suppose the crop contains `memory`. HGNetV2 converts its pixels into feature maps containing information about strokes and shapes. SVTR compares features at different locations through attention and other processing. The CTC head then predicts character probabilities along the horizontal sequence. None of these parts is manually told where the six individual letters are.

The NRTR branch also learns to read the labelled image during training. It uses an attention decoder with the correct preceding text supplied during training, and can attend to image features. The preceding **text characters** are different from preceding image patches. This branch gives an additional learning signal to the shared visual backbone.

At deployment, the recognition path is **crop -> backbone -> SVTR -> CTC -> text**. NRTR's training role does not mean an NRTR language model runs afterwards to correct the app's output. There is also no loss calculation during ordinary reading, because the app does not know the correct answer.

### Pooling and CTC positions

The original training backbone forces 40 horizontal feature positions. At inference, the number of positions follows width, approximately input width divided by eight.

At 64×1024 input, our adaptation uses approximately 128 positions in training and inference. These positions contain character/blank probabilities, not fixed annotated character boxes. CTC sums possible alignments during training and collapses repeated tokens/blanks during decoding. It needs enough positions for label characters and separators between identical adjacent characters.

Think of the positions as an ordered sequence of observations across the writing. At one position the model might favour `m`, at another it might favour a blank. Several neighbouring positions can favour the same character. CTC turns this longer sequence into a shorter piece of text. For example, `H H blank E E blank L blank L blank O O` decodes to `HELLO`: adjacent repeated predictions collapse, and blanks disappear. A blank between the two `L` predictions lets the decoder retain both letters.

Pooling controls how much spatial detail remains in that sequence. The key discovery was that making the input image wider did not automatically make the original **training** sequence longer: its pooling still forced 40 positions. A 1024-pixel-wide input could consequently train with 40 positions while producing about 128 at inference. We tested whether changing that training representation helped; we did not assume that more input pixels alone solved it.

P40/P128 both used labels fitting 40 positions. P40's failure therefore cannot simply be blamed on impossible long training labels. The experiment evaluates the effect of 40-position training versus width-proportional training **in this wide-input setup**. It does not prove ordinary 320-pixel-wide PaddleOCR training is inherently broken. Repeated-letter insertion is a measured diagnostic; the exact causal explanation should remain qualified.

### Guided training and loss weights

| Parameters | Default `use_guide=True` | G0 `use_guide=False` |
|---|---|---|
| Backbone | NRTR loss gradient | NRTR and CTC loss gradients |
| SVTR neck + CTC head | CTC loss | CTC loss |
| NRTR branch | NRTR loss | NRTR loss |

The default SVTR path detaches the backbone features before the CTC branch. CTC is still essential and trains the neck/head; it is not ignored.

There are two directions to keep separate. In the **forward pass**, image features travel through the network to produce predictions. During **backpropagation**, a loss produces gradients that tell parameters how to change. The guidance switch changes that second flow: with guidance on, CTC can use the backbone's features but cannot directly update the backbone through that branch. NRTR still updates it. With guidance off, both losses can update the backbone.

A loss considers probabilities as well as the final answer. Giving the correct letter a 90% probability can still produce a small penalty; giving it 10% produces a larger one. The optimiser uses gradients from the loss to change weights. The learning rate controls the size of those changes. This is why “low loss,” “correct text” and “high confidence” are related but not interchangeable measurements.

Both loss coefficients remain **1.0** in the completed S/G runs. They need not sum to one. With the guided branches updating separate parameter groups, Adam can largely cancel moderate constant gradient rescaling, so a simple coefficient sweep was deprioritised. This is not exact mathematical invariance in the real setup. With guidance off, both losses reach shared backbone parameters, so balancing their relative influence becomes a different question. No coefficient-sweep experiment is recorded as completed here.

## 6. Experiment history

The arrows below describe comparisons, not sequential checkpoint inheritance. Unless stated otherwise, each run starts independently from the original pretrained weights.

### A. Historical and data/recipe studies — completed by 6 November

Our first major batch asked what the **data and input recipe** contributed. We compared an untouched pretrained reader, an IAM-trained version, our own whiteboard data, longer lines and additional augmented copies. The version names identify those comparisons; they are not a chain where V3 necessarily resumed V2's checkpoint. Each relevant training run started independently from the same pretrained weights.

| Run | What it tests |
|---|---|
| V1 | Historical model, evaluated as deployed; preserve for history, not as the only fair baseline |
| Pretrained | Untouched PP-OCRv5_server_rec |
| V1b | IAM-only fine-tune using adapted recipe |
| V2short | Own-data recipe with labels of at most 40 characters |
| V3short | V2short plus pre-made augmented copies |
| V2 | Own data including longer lines |
| V3 | V2 plus pre-made augmented copies |
| C1 | Original-geometry recipe versus adapted recipe; changes several factors |
| P40 / P128 | Matched pooling-only pair at 64×1024, identical training lists restricted to labels fitting 40 positions |
| V4 | Name reserved for recognition plus preprocessing at inference; not a separate completed training run in these result packages |

All these training recipes also use PaddleOCR's built-in RecAug. “Without augmentation” in the 2×2 study means without **our extra pre-made copies**, not without any random augmentation.

Mean line CER (%), one seed per model:

| Model | CNN board | Held-out sentences |
|---|---:|---:|
| Pretrained | 45.79 | 30.93 |
| V1b, IAM fine-tune | 49.76 | 5.79 |
| V2short | 28.57 | 5.81 |
| V3short | 25.83 | 4.49 |
| V2 | 23.72 | 2.38 |
| V3 | 24.27 | 2.65 |
| C1 | 28.72 | 8.81 |
| P40 | 93.88 | 45.26 |
| P128 | 26.60 | 4.55 |

Before interpreting the table, understand CER. It counts character substitutions, deletions and insertions relative to the correct text. A CER of 25% means approximately one edit for every four reference characters under the reported averaging method; it does not mean only 75% of complete sentences are correct. A sentence can contain one wrong character and fail exact-line accuracy while still having a low CER. Our tables use mean line CER, which gives each crop equal weight; corpus CER instead divides all edits by all reference characters.

The own-data results showed that examples resembling our actual whiteboards matter. Including longer training lines also helped the measured sentence results. But we added more examples at the same time, so we cannot attribute every gain specifically to sentence length. The extra augmentation copies gave mixed results: a gain in the short-only setup did not automatically carry over to the longer-data setup.

The pooling pair gave a much larger difference. At the same wide input and on the same training examples, P40 was very poor on the CNN board, while P128 read substantially better. Even though those labels could fit within 40 positions, the representation used during training still mattered. This supports our adaptation at that geometry. It does not mean that the original 40-position design fails when used with its intended narrower input.

Interpretation:

- Own-data adapted fine-tuning substantially improves these whiteboard evaluations.
- Adding longer training examples helped in this experiment, but also increased the data by about 27–28%; it does not isolate length alone.
- Extra copies helped the short-only recipe in the measured seed but did not establish a benefit with the longer-data recipe. Repeat small gains before treating them as reliable.
- P40/P128 supplies strong evidence for the pooling adaptation at wide input. This pair is exactly matched in updates; the other data runs have approximately, not exactly, matched budgets because epochs were rounded.
- C1 changes geometry and multiple recipe details. Do not call it a pooling-only test.
- P128 had the lowest validation CER in this batch. V2's better test values do not justify selecting V2 using those test scores.
- The 64 uncertain-writer crops were present in the own-data V-runs. Do not call later evaluations of those old models on JH/DS definitely unseen-writer tests.

Evidence: [results](results_versions_2x2/results/report.md), [independent review](codex_update_20261107/REVIEW_BY_CODEX.md).

### A1. Additional historical-model readings on the Mac CPU

**Added 8 Nov by Codex, from Claude's proposed additions; CSV scores independently recalculated.** The [local CPU score file](board2notes_newwriter_test_20261107/scores_versions_2x2_models_mac.csv) contains all eight older models on the same 160 new-source crops. Recomputed mean line CER is:

| Model | New-source CER, Mac CPU |
|---|---:|
| V1b | 31.26% |
| V2short | 28.09% |
| V3short | 25.24% |
| V2 | 26.02% |
| V3 | 25.13% |
| C1 | 45.76% |
| P40 | 69.03% |
| P128 | 27.94% |

These are historical results with **writer overlap unknown**. V2's 26.02% here and clean S0's two-seed 26.79% are numerically close, but they differ in training recipe/history, seed and evaluation runtime. This is not an isolated removal-of-64-crops experiment. Similar aggregate errors cannot establish whether the old models saw those writers or rule out a contamination benefit. Keep the provenance caveat.

### B. Training strategies S0–S3 — completed 7 November

Purpose: test learning-rate and loss changes using a clean writer-exclusion baseline.

After the data and pooling batch, we wanted to investigate **how the model learns**, while keeping the data and working geometry fixed. Before doing that, we removed the 64 teammates' crops because their writers might overlap the intended new-writer evaluation. That produced S0: our clean reference recipe. It is “V2-style” because it uses our original short and long crops, but it is not the old V2 checkpoint with some files removed after training.

S1 asks whether early and later CNN layers should change at different speeds. Early layers already contain useful visual knowledge from pretraining, so we reduced their learning rates more strongly. The later layers and recognition components could adapt faster. This is a training strategy; the optimiser remained Adam and the network did not become deeper.

S2 asks whether easy training examples should contribute less to the CTC objective. Focal CTC reduces the influence of easy sequences relative to harder ones. It does not know whether a line is blurry or whether its label is wrong: a difficult line can have many causes. S3 combines S1 and S2 to check whether their effects add together.

We repeated each setup with two seeds. A seed controls random aspects of training, so two runs of the same recipe can learn different weights and make different mistakes. This is why a small improvement in one run is insufficient evidence that a setting reliably helps.

Common settings: **1,359 training entries; original pretrained start; 64×1024 input; width-proportional pooling; batch 16; Adam; base LR 1e-4; 300-step warmup and cosine schedule; RecAug; 71 epochs; exactly 6,035 updates; validation every 400 updates.** No pre-made augmented copies and no uncertain-writer teammates' crops. Ordinary and NRTR loss coefficients both 1.0.

| Run | Change from S0 |
|---|---|
| S0 | Reference recipe |
| S1 | Backbone stem/stage LR multipliers `[0.1, 0.1, 0.25, 0.5, 1.0]` |
| S2 | Focal CTC: sequence loss `L` becomes `L * (1 - exp(-L))^2` |
| S3 | Both changes |

Each setting has seed 1 and seed 2; `s2` denotes seed 2, not a second architecture. A new seed changes stochastic training and file order. Identical seeds do not guarantee bit-identical GPU execution.

Primary rule, declared before training: a setting helps only if its two-seed average validation CER difference has a 95% interval below zero **and** each seed improves against its matching S0 seed.

Mean line CER (%), average of two independent runs:

| Setting | Whiteboard val | CNN | New writers | Familiar-writer sentences | IAM |
|---|---:|---:|---:|---:|---:|
| S0 | 7.52 | 25.50 | 26.79 | 2.36 | 23.89 |
| S1 | 7.09 | 24.58 | 24.93 | 2.54 | 23.72 |
| S2 | 7.24 | 24.87 | 26.47 | 2.30 | 26.53 |
| S3 | 7.19 | 25.03 | 25.97 | 2.26 | 23.89 |

S1 looked most promising on the primary validation averages. Focal CTC was less convincing: it helped on the CNN evaluation in one seed and hurt in the other. Combining the changes did not establish an extra benefit over S1. These outcomes give us a reason to prioritise some ideas over others without pretending that every experiment improved the model.

The confidence interval explains why we did not simply declare S1 the winner. Our calculation repeatedly resamples groups of evaluation examples to see how much the measured difference varies. A difference interval entirely below zero supports lower error on that comparison. An interval crossing zero leaves improvement uncertain. It does **not** prove that the two methods behave identically. These intervals also do not capture every possible source of uncertainty, especially new writers and new training seeds.

Findings:

- None passed the primary rule. “Improvement not established” does not mean “no effect.”
- S1 has the lowest average validation CER and a promising two-writer gain, largely in short text.
- Focal CTC's direction changes with seed on important secondary sets. It does not establish a reliable benefit.
- S3 supplies no demonstrated additional benefit over S1.
- Focal weighting is per sequence, not specifically per blurry image or individual character. Large raw CTC losses receive weights near one, so the effect can be small.
- Same-recipe seeds differ by about 2.64 percentage points on CNN. This warns against overclaiming small single-seed improvements; it is not a universal noise threshold.

Evidence: [results](results_strategies_s0s3/results/report.md), [independent review](codex_strategy_review_20261107/ANALYSIS.md), [mechanism fact-check](codex_strategy_review_20261107/FACT_CHECK_CLAUDE.md).

### B1. Does the model fit its training examples?

**Added 8 Nov by Codex after checking the saved S0 log.** At the final logged update, S0 reports `norm_edit_dis=0.988813` and `acc=0.874999`. This shows strong performance on recent augmented training batches. However, PaddleOCR logs a median over a 20-batch window, and its normalised edit similarity uses a different denominator from our reported CER. **Do not label `1 - 0.988813` as a measured 1.2% full-training-set CER**, or say that exactly 87.5% of all training lines are correct.

The recent-batch fit, validation CER around 7.5%, familiar-writer sentence CER around 2.4%, and new-writer CER around 27% suggest that generalising to different handwriting is an important difficulty. The comparisons use different samples, conditions and selected checkpoints, so they do not isolate writer identity as the sole cause. Nor does fitting training batches prove that a different/larger SVTR could never help. It supports prioritising handwriting diversity and generalisation analysis before adding capacity without a specific diagnosis.

Evidence: [S0 training log](results_strategies_s0s3/runs/S0/train_log.txt), [G0 review](codex_guide_review_20261108/ANALYSIS.md). The recent-batch caveat follows inspection of `ppocr/metrics/rec_metric.py` and `ppocr/utils/stats.py` at the recorded PaddleOCR commit: normalised edit distance and the median training-log window are different from the report's full-set CER calculation.

### C. Guidance-off experiment G0 — completed 7 November, reviewed 8 November

Question: does letting the deployed CTC objective update the backbone improve whiteboard handwriting recognition?

This experiment followed our source-code investigation. We discovered that, in the existing guided setup, CTC trains SVTR and its prediction head while NRTR supplies the backbone's loss gradient. We therefore asked whether the objective used by the final reader should also help shape the shared visual features. G0 changes that gradient route, with the data, architecture size, loss coefficients and training budget held constant.

It was a hypothesis rather than an obvious fix. Two objectives updating shared features might provide useful supervision, but their preferred updates can also disagree. PaddleOCR's guided design deliberately separates those responsibilities. We needed the comparison to find out how the alternative behaves on our task.

- G0 versus S0, seed 1; G0s2 versus S0s2, seed 2.
- Same data in the same per-seed file order, pretrained start, configs, loss weights and 6,035 updates.
- Only training change: `use_guide=False` in the CTC branch's SVTR neck.
- Matching environment/source hashes and reference predictions support reuse of the saved S0 controls.
- Saved gradient probe confirms CTC-branch connectivity to the backbone when guidance is off.
- Saved export tests: 30/30 fixture predictions identical to each run's own checkpoint. This measures export equivalence, not 100% recognition accuracy.

| Evaluation | S0 average CER | G0 average CER | Interpretation |
|---|---:|---:|---|
| Whiteboard validation | 7.52% | 7.17% | Small, inconclusive improvement |
| CNN board | 25.50% | 25.70% | No clear difference |
| Familiar-writer held-out sentences | 2.36% | 2.41% | No clear difference |
| New writers | 26.79% | 22.77% | Promising reduction in both seeds and both writers |
| IAM | 23.89% | 27.15% | Worse in both seeds |

Primary validation difference: **-0.35 percentage points**, interval **-0.77 to +0.22**. Both seeds improve, but the interval crosses zero: **improvement not established**.

New-writer difference: **-4.01 points**, interval **-5.69 to -2.86**, approximately 15% relative error reduction. The two writers and all three length groups improve in the seed average; unfamiliar long text still has about 30% CER. These are exploratory findings because this set informed development, covers only two writers, and the intervals hold the trained seeds fixed.

**How much did the two seeds differ?** The absolute difference between seed 1 and seed 2 is shown below. These values are observed pair differences, not estimated limits on all possible training randomness.

| Evaluation | S0 seed difference | G0 seed difference |
|---|---:|---:|
| Whiteboard validation | 0.01 percentage points | 0.44 percentage points |
| CNN board | 2.64 | 0.49 |
| New writers | 1.19 | 3.98 |
| IAM | 0.57 | 4.43 |

G0 improves new-writer CER against its matching S0 seed in both runs, but the gain is 2.62 points in seed 1 and 5.41 points in seed 2. The direction repeats; the size varies substantially. Its photo-bootstrap interval does not include uncertainty from training additional seeds. Do not interpret the mean -4.01-point result as a guaranteed improvement for the next run or writer.

G0's best checkpoint is epoch 15; G0s2's is epoch 19. Both completed 71 epochs. Lower late-training errors do not prove later checkpoints generalise better.

Decision: retain G0 as a fresh-writer confirmation candidate; do not declare it an established overall winner. S1 remains best on the average primary validation metric. IAM degradation is consistent with a domain trade-off, but does not prove its internal cause.

In everyday language, **G0 helped the two unfamiliar handwriting styles we measured, but it did not improve every kind of handwriting**. Familiar-writer sentence performance was almost unchanged, and paper handwriting became worse. Its validation improvement was also too small and uncertain to pass the decision rule agreed beforehand.

One concrete regression is the short-word crop JH33.1, labelled `Tough`: S0 seed 1 reads `Tos`, G0 seed 1 returns an empty string, and G0 seed 2 reads `Touh`. Add **empty outputs and missed faint/short words** to the error-review checklist. This individual example establishes that dropouts occur, not that faintness is their proven cause or that G0 drops more words overall. Compare blank-output counts and the actual crop pixels before making that broader claim.

That is useful model-development evidence. We changed a specific learning connection, checked that the connection really changed, and measured both benefits and drawbacks. However, we have already used the current new-writer set during development. The next confirmation should use fresh writers, so we do not keep choosing methods using the same examples and then call those examples an untouched final test.

Evidence: [G0 results](results_guide_g0/results/report.md), [merged comparison](results_guide_vs_s0/report.md), [independent audit](codex_guide_review_20261108/ANALYSIS.md).

### D. Fresh Jy evaluation and OCR prototype — completed 8 November

The user confirmed that Jy should be unseen relative to the previous data and that a teammate reviewed the crop labels. We froze the original-crop inventory and evaluated all 298 originals from both supplied folders; no synthetic variants entered the main score and no model was retrained. We used both seeds of S0, S1 and G0 rather than select a favourable seed using Jy.

The confirmation rule was saved before prediction: G0's average CER change versus S0 must have a photo-cluster 95% interval below zero, and both paired seeds must improve. S1 was a secondary comparison. This is a new collection-level confirmation, not a revision of the old validation primary criterion.

| Recipe | Seed 1 CER | Seed 2 CER | Average CER |
|---|---:|---:|---:|
| S0 | 7.35% | 7.33% | 7.34% |
| S1 | 5.57% | 5.48% | 5.52% |
| G0 | 5.91% | 5.38% | 5.64% |

- G0 versus S0: **-1.70 percentage points**, interval **-3.02 to -0.33**, both seeds improve. **Passes the fixed confirmation rule on Jy.**
- S1 versus S0, secondary: **-1.82 points**, interval **-3.01 to -0.98**, both seeds improve.
- S1's average is only about 0.12 points lower than G0's. This does not establish that S1 is superior to G0.
- All six produce zero blank readings on these original crops.
- The collection is dominated by individual words (241/298). Report the 57 multi-word crops separately and do not generalise the overall number to distant long sentences or all writers.
- Source-photo bootstrap holds the trained seeds fixed and does not estimate population-wide writer uncertainty. The writer count/IDs still require provenance documentation.
- Six runs × 298 crops = 1,788 predictions. The root review independently recalculated every edit count/CER; all six caches share the same backend implementation hash.
- This uses local Paddle 3.3.1 CPU. Differences from previous Linux GPU scores can include runtime differences; comparisons within this run use the same implementation.
- Inference seconds were collected alongside other OCR work and are not controlled speed benchmarks.

The full-photo path also ran on all 16 Jy photos using S0, S1 and G0s2. It saves **the same 301 detected regions/crops** for every reader. These outputs show what the app actually reads, including mistakes, but they do not yet establish complete-photo transcription accuracy. Existing crop labels omit some visible writing, and full-photo geometry/order were not originally supplied. A separate assistant visual review recovered positions for all 298 supplied targets; annotation scope/completeness remains to be confirmed.

The [spatial diagnostic](whiteboard-ocr/outputs/jy_ocr_evaluation/full_photo_diagnostics/report.md) associates predictions to those supplied targets by polygon overlap, never by whether the OCR text is correct. Its heuristic graph links all 298 targets, identifies one merge candidate and retains four unmatched predictions separately. Component-level annotated-target CER is S0 **6.36%**, S1 **5.11%**, G0s2 **5.38%**. These are micro character scores over matched components/missed supplied targets, not the earlier crop-mean scores and not certified full-page CER. Unmatched predictions are excluded from this component CER because the label set is incomplete; they are reported explicitly. The optional ordered-all-output diagnostic also exposes scope/order problems. Do not describe the 298/298 spatial association as perfect standard detector recall.

An OCR-only local prototype is now implemented with JPG/PNG/HEIC upload, model selection, detected-region overlays, text display and text/JSON download controls. Actual CLI and browser tests read Jy Photo 23 as nine regions in approximately three seconds after loading on this machine. This is one sample, not an overall latency benchmark. There is no VLM/LLM or slide generation in this version. The browser automation's download-event check timed out, so the download controls are implemented but a completed file download was not confirmed by that test.

Open the local app at `http://127.0.0.1:8501` while the server is running; restart using `Assignment/run_ocr.command`. This is a local prototype, not completed cloud deployment.

Evidence: [Jy result report](whiteboard-ocr/outputs/jy_ocr_evaluation/report.md), [notebook 11](whiteboard-ocr/notebooks/11_jy_ocr_evaluation.ipynb), [full-photo previews](whiteboard-ocr/outputs/jy_ocr_evaluation/full_photos/review.html), [OCR prototype instructions](whiteboard-ocr/ocr_prototype/README.md).

Notebook 11 was executed end to end after its saved-photo timing-field compatibility fix: all nine code cells completed without saved errors. It defaults to displaying saved results, crops and photo diagnostics; fresh inference is controlled by explicit switches.

### E. Student-facing OCR flow — added by Codex, 8 November

The first local interface above exposed research model names and technical output. Following the user's UX review, it was replaced with **Add photo → Check photo → Review notes**. Students upload or try a sample, optionally rotate/trim the photo, extract text, compare it with the photograph, edit it and copy/download notes. Region boxes are optional and a close-up tool helps inspect original letter strokes. There is no student model selector, admin page or technical live demonstration; the latter is deferred for discussion.

The provisional integration default is **S1 seed 1**, using the original validation result as the selection basis. This does not establish S1's superiority to G0 and does not alter any experiment verdict. The backend and saved comparison outputs remain unchanged. Notes survive returning to the photo tools; explicit consent is needed before overwriting edits, and starting another note offers save/cancel/confirm. Drafts remain in the browser session, not a persistent note library.

Verification: seven image-helper tests and thirteen mocked Streamlit journey tests pass, including preserved crop settings/pixels across navigation. A real Photo 23 upload and sample extraction ran with S1; the browser showed nine regions. An OCR error was edited and the exact corrected text was verified in the clipboard. Actual cancel/confirm dialogs and edit preservation across adjustments worked. The close-up and phone-sized layout were exercised. Download preparation worked, but the in-app browser's download-event test did not confirm a completed file on disk; check that in the presentation browser. No user study has yet established "outstanding usability". Next UI evidence is a few students completing the task without coaching; cloud deployment remains pending.

Evidence: `whiteboard-ocr/ocr_prototype/README_UI.md`, `student_app.py`, `student_flow.py`, `tests/`, and the student-flow screenshots in `whiteboard-ocr/outputs/jy_ocr_evaluation/`.

Follow-up on 8 November: the user requested visible computer-vision evidence within the student task. Detection polygons now appear by default and are clickable. Selecting a region highlights it in orange and synchronises the selector, exact extracted crop and original OCR prediction, preserving editable notes. An expandable **How your photo was read** view shows the actual input, region overlay and crop-to-text result. This is visibility of the existing pipeline, not a new model-performance claim. The real sample's region 4 was clicked and its original `tich` reading stayed separate from the manually corrected `rich` notes. The automated suite now has 21 passing checks. Technical model-comparison/admin demonstration remains deferred.

## 7. Current decisions and open questions

Full-photo evaluation follow-up, 8 November: `notebooks/12_full_photo_evaluation.ipynb` now contains a separate saved-results evaluation of all 16 Jy photographs. The assistant visually checked all originals, retained the 298 supplied labels and added faint `laughing`/`gas` in Photo 14, producing 300 reference targets. Complete-photo text/order remains pending dataset-owner approval; scores are provisional, not certified final accuracy. `outputs/jy_ocr_evaluation/full_page_evaluation/review.html` presents all references beside originals without predictions.

Using all returned text in actual app order, page corpus CER is S0 27.92%, S1 27.52%, G0s2 27.62%. GT-assisted geometric reordering yields 6.72%, 5.64% and 5.84% respectively. The latter is a diagnostic requiring answer-key geometry, not something the app achieves. The largest gaps occur on column layouts in Photos 8/14/15/19. The chosen column-major convention should be confirmed because unrelated word lists can admit multiple reasonable orders. A cross-column merge in Photo 8 and two printed-sign outputs beside the board in Photo 14 remain inspection examples. All 300 references have heuristic geometric overlap, which is not standard detector recall. No inference/training/app changes occurred. Five notebook code cells executed without errors and 48 scores were independently checked. Next is owner confirmation, then focused layout/grouping development with fresh-photo confirmation after tuning.

| Decision | Current position | Reason |
|---|---|---|
| Sequence geometry | Keep the tested width-proportional pooling in the adapted reader | Large matched-pair result at wide input |
| Formal strategy reference | S0 | Clean data and common controls |
| Candidate training strategies | S1 and G0 | Existing validation/new-writer evidence plus a positive fresh Jy comparison for both; no direct superiority established |
| Focal CTC | Do not promote as a demonstrated improvement | Inconsistent seeds and no primary-rule success |
| CTC/NRTR coefficient sweep | Not completed; no automatic expansion now | Must account for gradient routing and Adam; specify a hypothesis first |
| Larger/deeper SVTR | Not tested in these batches; lower priority | No demonstrated capacity bottleneck; good training fit suggests examining generalisation first |
| SR | Paused | Tested versions did not reliably improve OCR |
| Perspective | Conditional experimental component | Geometry selection can fail; do not apply indiscriminately |
| Preprocessing V4 | Still requires final-reader full-photo evaluation | Crop-level recognition comparisons do not establish full-photo performance |
| Final deployed recogniser | Not frozen in this record | Select using an explicit policy, then verify its real runtime |

Good training fit does not prove a larger model can never help, and a failed G0 would not disprove a useful SVTR change. Guidance routing and feature-processing architecture test different questions.

## 8. What the evidence does not establish

1. A two-seed average is not a deployable ensemble. Select a real exported checkpoint or explicitly implement/evaluate an ensemble.
2. Averages can hide regressions. Inspect both seed results and example failures.
3. Crops, photos, phrases, boards and writers are different units. More crops of the same seven phrases do not provide seven hundred independent texts.
4. Validation shares source photos with training for different texts, and also selects checkpoints. Its value is not an untouched final-test estimate.
5. Bootstrap intervals over photos/sentences do not measure uncertainty across a population of writers or future training seeds.
6. Repeatedly using a test to design changes turns it into development evidence. A final generalisation claim needs a fresh held-out set.
7. Reader results on supplied line crops do not test missed detections, merged boxes, reading order or full-photo transcription quality.
8. Geometry/SR/old PP-OCRv6 scores and new PP-OCRv5 training scores are not automatically comparable.
9. The pooling and training strategies are adaptations/investigations of existing mechanisms. Describe our contribution without claiming invention or a guaranteed rubric score.

## 9. Next actions

These are proposed actions, not completed work. Add owners and dates when agreed.

| Priority | Action | Completion evidence | Owner/status |
|---|---|---|---|
| 1 | Confirm label-review coverage and writer provenance | Final manifest with writer IDs, review status and split lineage | To assign |
| 1 | Confirm G0 versus S0 on fresh handwriting | Fixed-rule six-reader Jy run completed with both seeds; G0 improves on this collection | Completed 8 Nov; broader writer claims remain limited |
| 1 | Select and freeze one recogniser for integration | Model folder/hash, exact preprocessing/resize/dictionary and measured validation criteria | To agree |
| 1 | Run full-photo comparison | Sixteen-photo baseline predictions with fixed detections now saved; scope/annotation review and optional-preprocessing comparisons remain | In progress |
| 1 | Connect OCR -> structured JSON -> slides | A saved end-to-end example and documented field contract | Team to update |
| 1 | Demonstrate local and cloud app | Local OCR prototype tested; cloud deployment, final app integration and controlled runtime measurements remain | Local prototype completed 8 Nov; cloud pending |
| 2 | Measure optimisation/scalability | Appropriate runtime/concurrency/model-size comparison on actual target hardware | Pending |
| 2 | Collect more writers if time permits | New labelled text from recorded writers; no final-test writer enters training | Proposed |
| 2 | Check empty outputs and short/faint-text regressions | Matched crop examples and blank-reading counts across seeds; distinguish blank recognition from a missed detection | Proposed |
| 2 | Diagnose P40 inference pooling without retraining | P40 validation predictions using its usual approximately 128 inference positions versus forced 40; record the changed inference code and fixed scoring policy | Proposed; checkpoint availability on RunPod is team-reported |
| 3 | Further architecture/loss experiments | Specific diagnosis and fair comparison defined first | Optional; not automatically scheduled |

Two or three fresh writers with a few dozen crops can provide a practical confirmation, not a population-level claim. Record both writer and photo identities. Do not spend the remaining project period on an unbounded model search at the expense of integration.

The P40 inference diagnostic asks whether matching the training sequence length at reading time reduces its failure. It requires no new training, but forcing 40 positions also changes spatial compression during inference. A recovery would support the mismatch explanation without proving it is the only cause. An optional four-cell diagnostic could evaluate both P40 and P128 with both inference poolings. Fix the analysis on validation first. Merely setting the existing **training-only** pooling switch is insufficient: verify that the evaluation path really produces 40 positions. No such diagnostic has been run as part of this document update.

## 10. Where the evidence and models are

Paths below are relative to `Assignment/` for portability.

| Artifact | Location |
|---|---|
| This record | `PROJECT_PROGRESS.md` |
| Visual results notebook | `whiteboard-ocr/notebooks/10_training_results_visual_explorer.ipynb` |
| Fresh Jy evaluation notebook | `whiteboard-ocr/notebooks/11_jy_ocr_evaluation.ipynb` |
| Fresh Jy predictions and report | `whiteboard-ocr/outputs/jy_ocr_evaluation/` |
| Local OCR-only prototype | `whiteboard-ocr/ocr_prototype/`; launch with `run_ocr.command` |
| Data/pooling results | `results_versions_2x2/` |
| LR/focal results | `results_strategies_s0s3/` |
| Guidance-off results | `results_guide_g0/` |
| G0-versus-S0 merged report | `results_guide_vs_s0/` |
| Exported models | Each result package's `results/models/` |
| Run configs and logs | Each result package's `runs/<run>/` |
| Exact scripts used for training | Each result package's `source/b2n_versions.py` |
| Dataset ZIPs | `board2notes_versions_runpod/b2n_versions_data.zip`, `b2n_newwriter_data.zip` |
| Model/data/recipe review | `codex_update_20261107/REVIEW_BY_CODEX.md` |
| S0–S3 independent review | `codex_strategy_review_20261107/` |
| G0 independent review | `codex_guide_review_20261108/` |
| Earlier SR/fusion history | `whiteboard-ocr/docs/SUPER_RESOLUTION_RUNS.md` |

Notebook 10: earlier sections cover data/pooling/CTC; **section 7b** covers S0–S3/G0, seed comparisons, writer/length groups and image examples. Use saved readings as evidence; they do not rerun the deployed model.

### Reproducibility and RunPod storage

- Completed S/G runs record PaddleOCR commit `91279fee56ef6eed096c832d3912abdfc4d2725e`, repository patch hash, dataset/pretrained hashes and Paddle GPU 3.2.0 on Python 3.12.
- RunPod persistent Python environment: `/workspace/venv_b2n`; repository: `/workspace/b2n/PaddleOCR_versions`; experiment folders: `/workspace/b2n/experiments/<experiment>/`.
- Persistent packages can be reused on compatible pods after runtime checks; preserving the volume is essential. Stopping/terminating a pod is different from deleting its network volume.
- Network filesystem `df` output can show shared filesystem capacity rather than our quota. Check the provisioned volume quota before training.
- Downloaded inference models are enough for inference, but do not assume full optimiser/resume checkpoints are inside a results ZIP. Some runs intentionally removed optimiser/resume copies after completion.
- A saved 30-crop export check verifies those fixtures in that runtime. Test the actual local/cloud deployment runtime too.
- Preserve originals and experiment identities; use a new experiment name for changed settings.

### Local CPU versus RunPod GPU — evidence status

**Claude-reported on 8 Nov; comparison not independently verified in this update.** Claude reports that V2's exported model produced 157/174 identical CNN-board readings between a Mac VM CPU and RunPod GPU, with mean CER 23.78% versus 23.72%. The supplied scorer path is `~/b2n/score_static_mac.py` **inside the reported VM environment**. That script and the per-crop CPU/GPU comparison output were not located at the corresponding host-Mac path during this review. The new-writer CPU score CSV above is available, but it cannot verify this separate CNN comparison.

Retain this as a reported lead and add the actual script/output, model hashes, resize/dictionary settings and runtime versions before using the exact figures as verified deployment evidence. CPU/GPU readings can differ; the deployed runtime should be tested directly. No fresh inference was launched for this documentation update.

### RunPod storage and housekeeping — reported snapshot, not live state

Claude reports the following cleanup on 7 Nov. No live RunPod inventory was checked during this update, and the older 7 Nov handover only documents an earlier snapshot/planned deletion. The current remote state therefore remains **team-reported**:

| Reported item | Status to preserve/confirm |
|---|---|
| `versions_2x2` reduced from about 21 GB to 4.6 GB | Resume/optimiser state and `best_model/` reportedly removed; all `best_accuracy.pdparams` reportedly retained |
| `versions_v3` | Reportedly removed |
| `ab_v1`, about 9.3 GB | Reportedly contains a `results.zip` not yet downloaded; verify local backup before any deletion |
| Old RunPod API keys | Housekeeping request: review and revoke obsolete keys in the appropriate account/settings |

These are handover items, not commands or permission to delete anything. Do not put API key values in this document. This update did not access, reveal, revoke or delete any credentials or remote files. Confirm checkpoint availability before scheduling the P40 diagnostic.

## 11. Update template for Claude and teammates

Copy this block after the chronological log for each new result:

```text
Date / author:
Workstream and experiment ID:
Question or problem:
Baseline and starting checkpoint:
Exact change:
Data version, writers, splits and label-review status:
Training budget / seeds / runtime (if relevant):
What stayed identical:
Primary metric and decision rule (declared when?):
Results, including failures and uncertainty:
Decision and its limits:
Files / model hashes / evidence:
Next action, owner and status:
```

### Chronological log

| Date | Event | Record |
|---|---|---|
| Earlier project work, mainly October | Dataset annotation, preprocessing, paired SR, text-prior restoration and multi-photo reading investigations | Historical notebooks and SR/fusion record |
| 5 Nov 2026 | Professor feedback shifted emphasis toward adapting PaddleOCR internals; own-data inventory and training package work | Discussion, briefing and prepared dataset records |
| 6 Nov 2026 | Eight data/recipe/pooling training runs completed | `results_versions_2x2/` |
| 7 Nov 2026 | Data/pooling merge and results independently reviewed; provenance clarification motivated clean S/G baseline | `codex_update_20261107/` |
| 7 Nov 2026 | S0–S3 trained with two seeds; no strategy passed the primary rule | `results_strategies_s0s3/`, independent review |
| 7 Nov 2026 evening | G0/G0s2 completed; gradient routing and exports checked | `results_guide_g0/` |
| 8 Nov 2026 | Independent G0 audit confirms promising new-writer result and inconclusive primary validation; notebook 10 section 7b available | `codex_guide_review_20261108/`, notebook 10 |
| 8 Nov 2026 | This master project record drafted for team additions | `PROJECT_PROGRESS.md` |
| 8 Nov 2026 | Codex checked Claude's proposed additions: label-review record, CPU score CSV, seed spread and empty-output example added; runtime/storage claims marked pending evidence | This document; linked review files and result records |
| 8 Nov 2026 | Jy collection inspected: useful fresh originals/full photos; supplied augmentation-family split leakage documented; recommended reserved evaluation after label/provenance review | `codex_jy_dataset_review_20261108/` |
| 8 Nov 2026 | Teammate crop-label review confirmed by user; all six frozen readers evaluated on 298 originals; G0 passes this new confirmation rule and S1 improves in secondary comparison | `whiteboard-ocr/outputs/jy_ocr_evaluation/` |
| 8 Nov 2026 | Sixteen full-photo predictions generated with identical detected boxes for three readers; local OCR upload prototype tested in browser | `whiteboard-ocr/ocr_prototype/`, full-photo preview gallery |
| 8 Nov 2026 | Student OCR flow replaces technical model controls; editable notes, photo tools, close-up review, safe navigation and export actions tested | Prototype README_UI, 20 automated checks and student-flow browser screenshots |

### Requested additions from Claude/team

- Confirm actual contributors and owners for every workstream.
- Add current app/UI, cloud/local deployment and scalability status with evidence.
- Locate/link the current pptxGenJS converter and LLM interface; the earlier `pptx-demo` handover was team-reported and that exact folder was not located during this draft.
- Add the latest notebook 08/09 diagnostic outcome and intended deployment preprocessing settings; mark the reader used.
- Confirm remaining training-label review coverage and writer provenance; the 160 new-writer and 57 held-out evaluation-label review is now documented above.
- Supply the Mac-VM CPU/GPU comparison script/output and current RunPod storage inventory before treating the reported exact figures as independently verified.
- Record the final chosen model and why it was selected, without replacing the fixed primary-rule verdicts.
- Update any older notes recommending a loss-coefficient sweep to reflect the subsequent gradient-routing and Adam discussion; preserve those notes as history rather than treating them as the active plan.

## 12. Revision and explaining the work aloud

### A short narrative you can practise

“Our project converts handwritten whiteboard photographs into structured notes and slides. At first, we focused on collecting and labelling data and improving the photographs through perspective correction, cropping, enlargement and other preprocessing. We also investigated super-resolution, but sharper-looking images did not reliably reduce OCR errors in our tests.

“After the professor asked us to investigate PaddleOCR more deeply, we examined its recogniser. We found that the original training pooling forced 40 sequence positions even when our wide input produced approximately 128 at inference. We adapted the pooling and tested a matched pair. That produced a large difference in the tested wide-input setting.

“We then fixed the data and geometry and compared training strategies. We tried different learning rates across backbone stages, focal CTC and their combination, each with two seeds. Layer-wise learning rates were promising, while focal CTC was inconsistent. None passed our primary validation rule.

“Finally, we investigated gradient routing. The original guided setup prevents CTC loss from directly updating the backbone. We allowed that update in G0. Error on the two evaluated unfamiliar writers fell from about 26.8% to 22.8%, with improvements in both seeds, but validation improvement remained inconclusive and IAM performance worsened. We therefore regard it as a candidate needing fresh-writer confirmation. The remaining work is to freeze the reader, evaluate the complete photo pipeline and finish local/cloud integration.”

This is a revision script based on the current evidence. Add teammates' verified integration work before using it as the complete team presentation.

### Questions to check your understanding

Try answering each question before reading its answer.

1. **Why do we need labels as well as crops?** The crop is the input; the transcription tells the recogniser what answer it should learn. A detector additionally needs locations on full images.
2. **What is a baseline?** A defined reference setup against which we compare a change. S0 is our clean reference for the S/G strategy studies.
3. **What is a held-out sentence?** A sentence excluded from training, including its related photos and derived augmentation. Its writer can still be familiar.
4. **What does SVTR do, and have we trained it?** It processes visual-feature relationships before CTC predicts characters. CTC loss already trains it in every S/G run.
5. **What are 40 or 128 positions?** Horizontal feature-sequence positions where the model predicts character/blank probabilities. They are not 40 or 128 manually cropped letters.
6. **Why did we test P40 against P128 separately?** To measure the pooling change using the same examples, wide input and training budget, instead of mixing it with a data change.
7. **What is the difference between a loss and an optimiser?** The loss measures prediction error and produces gradients. The optimiser uses those gradients to update parameters.
8. **Why train with two seeds?** To see whether a measured gain repeats under another training run, rather than trust one lucky outcome. Two seeds still provide limited evidence.
9. **Does guidance on mean CTC does not learn?** No. CTC still trains SVTR and its head; its gradient stops before the backbone. NRTR supplies backbone supervision.
10. **What exactly did G0 change?** It allowed CTC gradients into the backbone. It did not add SVTR layers or change Adam, the data or the loss coefficients.
11. **Why is G0 promising but not a confirmed overall winner?** The unfamiliar-writer gain repeats, but covers two development writers; the primary validation interval crosses zero and other sets do not all improve.
12. **What does “export verified” mean?** The exported model and its checkpoint produced identical readings on the checked fixtures. It does not mean every reading was correct.
13. **What counts as finishing the project?** A justified model choice plus a working photo-to-slides system, reviewed data, local/cloud deployment, UI and measurements addressing the rubric. More experiments alone do not complete those parts.

### When adding a new experiment, explain its place in the story

Write one paragraph beginning with the observed problem, then describe the proposed cause, the single change being tested, the result and the decision. Keep observations separate from interpretations. For example: “G0 worsened IAM CER” is an observation; “G0 became more specialised” is a possible interpretation. That habit makes the record easier to understand and the final report easier to defend.

## Added by Codex — 9 November: diagnosing remaining recognition errors

After the user agreed to investigate generalisation, Codex recomputed 10,806 saved readings from S0/S1/G0 and their second seeds over 1,801 unique crops, collected image metrics for 825 whiteboard-related crops, and visually inspected 53 deliberately selected error/control examples. No model inference, training or original-label changes occurred. Notebook [13_recogniser_generalisation_review.ipynb](whiteboard-ocr/notebooks/13_recogniser_generalisation_review.ipynb) has five executed code cells and no errors; [the audit report](codex_generalisation_audit_20261109/ANALYSIS.md) links the source hashes, measurements, image gallery and provisional visual notes.

The audit finds 66/160 JH/DS crops with at least 15% CER in all four S1/G0 checkpoints, versus 22/298 Jy crops (a post-hoc inspection threshold, not a formal test). Missing characters account for about 50% of S1's and 63% of G0's aligned JH/DS edits; substitutions dominate Jy. Whole-word empty readings are rare. Several difficult crops have faint writing or large margins, while readable unfamiliar sentences also fail. This supports a generalisation/input-robustness investigation, not a demonstrated backbone-capacity bottleneck. Simple crop-height/contrast/sharpness subgroup associations remain confounded.

Three crop/answer-key questions are flagged separately: `lecture_01_124.png` appears to show “which class wins” rather than the supplied “which is right”; `lecture_04_012.png` has multiple writing regions and unclear target scope; `ds__train__3.2.3.jpg` may have a visible period missing from the label. These are assistant observations awaiting independent confirmation; the existing labels and benchmark scores remain unchanged.

**Proposed next diagnostic, not yet run:** freeze the existing S1 seed-1 app reader and compare original crops with manually marked, safely padded writing-region crops, including low-error controls. Define boxes from images before viewing new predictions, retain all target strokes, and measure whether deletion errors improve specifically on large-margin cases. This tests crop-margin/stroke-scale sensitivity without new training; manual boxes are an oracle diagnostic, not automatic deployment. Only after that result should a practical crop policy or a targeted scale/margin training strategy be selected. Both JH/DS and Jy now inform development, so reserve fresh writer/photo groups for final confirmation.


## 9 Nov 2026 update (Claude): crop-margin diagnostic — completed

Author/date: Claude, 9 November 2026. Evidence: [`claude_margin_diagnostic_20261109/MARGIN_DIAGNOSTIC.md`](claude_margin_diagnostic_20261109/MARGIN_DIAGNOSTIC.md), frozen design `PROTOCOL.json`, per-crop results `RESULTS.json`.

- **Question:** do large blank margins (writing shrunk by the resize to 64 px) contribute to JH/DS errors, especially deletions?
- **Design (fixed before inference):** all 160 JH/DS crops; writing-region boxes from images only (ink-mask rule plus a visual check of every changed box, 4 manual exceptions); frozen S1 seed-1 export read on the Mac CPU for original and tightened crops; groups by how much height the box removes. Jy excluded (already tight; the rule missed light strokes there).
- **Result: supported under the pre-declared rule.**
  - Large-margin crops (box keeps ≤ 60% of the height, n = 51): CER 35.4% → 6.3% (-29.1 points, 95% CI -36.2 … -19.7); deletions 323 → 24; 40 better / 2 worse.
  - Controls (≥ 85% of the height kept, n = 48): -5.1 points. Difference in differences: -24.0 (CI -31.7 … -13.2).
  - All 156 non-flagged crops: 23.7% → 10.1%.
- **Interpretation:** much of the JH/DS "unfamiliar-writer gap" is a crop-scale effect of how those test crops were cut, not handwriting style alone. A smaller real gap remains (about 10% tight JH/DS vs about 2.4% familiar sentences and 5.5% Jy). Earlier new-writer numbers (S0/S1/G0 on original crops) remain valid as measured but should be quoted with this caveat; G0's exploratory new-writer gain may partly reflect robustness to small text. Not yet tested.
- **Limits:** oracle boxes (not an app feature); JH/DS already development data; one reader; CPU scorer (114/160 readings identical to the saved GPU readings, same mean CER).
- **Proposed next (not done):** (1) read the tight crops with S0 and G0 seed 1 (no training); (2) measure the margins of the detector's crops in the full-photo path, and only if they are loose, test an automatic ink-tightening step on full photos; (3) optionally, scale/margin augmentation in training with matched seeds.

| Date | Event | Record |
|---|---|---|
| 9 Nov 2026 | Crop-margin diagnostic (frozen S1 seed 1, 160 JH/DS crops): supported; large-margin CER 35.4% → 6.3% | `claude_margin_diagnostic_20261109/` |

**9 Nov follow-up (Claude):** the same frozen crops read with S0/S0s2/S1/S1s2/G0/G0s2 on the Mac scorer. Tightening helps every reader by about 11–15 CER points, with deletions down about two-thirds. G0 vs S0 (mean of seeds) shrinks from -3.26 (-4.53 … -2.15) on original crops to -1.65 (-3.08 … -0.52) on tightened crops: a smaller advantage remains (clear in seed 2, not seed 1). S1 vs S0 on tightened crops is not clear. 26 crops remain hard for all six readers, mostly already-tight unfamiliar sentences. Evidence: `claude_margin_diagnostic_20261109/RESULTS_MODELS.json`, notebook 14 section 7. No training or app change.

**9 Nov follow-up (Claude): full-photo detector comparison (notebook 15).** On the 16 Jy full photos, `PP-OCRv5_mobile_det → S1` (the app) was compared with `PP-OCRv5_server_det → S1`. Everything after detection was identical: app thresholds, grouping, ordering, cropping and the S1 export. The run used the Mac app environment (Paddle 3.3.1); the provisional answer keys were not owner-approved, and the decision rules were frozen before any server output was opened.

- **Server detector:** not worth switching.
  - Page CER 27.5% → 25.6% (-1.96 points, photo-bootstrap CI -6.4 … +0.6). All of the gain is in Photo 14 and comes from a reading-order threshold coincidence, not better boxes; excluding Photo 14 the two are tied (822 vs 827 errors).
  - Both detectors cover all 300 targets with 0 missed and the same 2 extra regions.
  - Cost: detection 11.9× slower (5.8 vs 0.49 s/photo), app path 10.8 vs 4.9 s/photo, load 17.4 vs 3.0 s, peak memory 9.9 vs 1.1 GB.
- **App crops:** no excess-margin problem.
  - 0% of app crops have large margins, against 32.5% of notebook 14's JH/DS dataset crops; median ink height in S1's input is 47/64 px.
  - The same targets read from annotation boxes and from mobile crops differ by +0.08 points (CI -0.46 … +0.67).
  - Crop tightening is not indicated for the app.
- **Where the error comes from (mobile):** recognition floor 4.7 points; boxes, grouping and crop +0.9; **reading order +21.9**.
  - `spatial_order` misses the columns on Photos 8, 14, 15 and 19, which sit at about 74% page CER each. Their column gaps (2–59 px) are below the 1.5 × line-height cut threshold (53–74 px).
  - One cross-column grouping merge occurs in Photo 8.
- **Proposed next (needs approval, app change):** evaluate reading-order and column rules offline with the same page-CER scorer. GT-reordered CER (about 5.6%) is the ceiling.

Evidence: `whiteboard-ocr/outputs/detector_comparison/` (PROTOCOL.json, per-arm run.json and per-photo JSON, crops, `analysis/RESULTS.json`, `analysis/INTERPRETATION.md`); `whiteboard-ocr/scripts/run_detector_comparison.py` and `detector_comparison_analysis.py`; notebook `whiteboard-ocr/notebooks/15_full_photo_detector_comparison.ipynb`. No training, no app change, no crop tightening.

### 9 November — user decision: defer reading-order changes

The user explicitly chose to record the reading-order limitation and defer further column/order development because the intended note structure is not yet defined. Do not proceed with the previously proposed reading-order or cross-column-grouping changes under that earlier plan. Keep the existing OCR application unchanged for now; deferral does not establish that its ordering is correct.

Codex independently reproduced the saved page CER (mobile 27.52%, server 25.56%) and the photo-bootstrap comparison interval, and verified that all three comparison arms used the same recorded S1 reader identity. The overlap diagnostics and large-margin counts reproduce. Claims remain limited to these 16 provisionally annotated Jy photos: heuristic overlap is not perfect detector recall; the CER ladder is approximate; answer-key-assisted reordered CER is not a guaranteed achievable score or a hard ceiling. The protocol records that its rules were frozen during execution before server output was inspected, not before inference began. The runner hash matches the protocol; the current analysis-script hash differs and that provenance change needs documenting.

Next recommended recognition-side investigation (not started by this entry): review the remaining difficult JH/DS cases after tightening, using the six saved model readings from notebook 14. Separate ambiguous labels/crops, degraded input and readable character confusions before selecting another training or architecture experiment. This returns to the generalisation investigation after accounting for the large crop-margin confound. No new training, detector switch or automatic tightening has been selected.

**9 Nov follow-up (Claude): review of the 26 persistently difficult tightened crops (notebook 14, section 8).** 18 of the 26 fail because the writing reaches S1 compressed.
- 16 are JH sentence lines written on a 3–7° slope and cut with an upright box: the box is about 3× the writing band, so after resizing to 64 px the text covers only the first 35–50 of S1's 128 output positions (the rest lies over padding) for 38–60 characters, and all of them emit their sentence ending as garbled text inside the padding.
- 2 are DS lines whose box kept a blank band.
- The rest: 5 short-word letterform confusions shared by all six readers, 1 faint word, 1 rotated word and 1 legible long line.
- Three possible label mismatches are flagged for owner check (JH6.6 "access", DS 1.4.2 "They", JH30.2 "Giant"); labels unchanged.
- Recommended next (not run): rectify long sloped lines (label-free angle from the ink mask, rotate, re-tighten) and re-read with the six frozen readers under a pre-declared rule.

Evidence: `claude_margin_diagnostic_20261109/persistent_review/`. No training or app change.

**9 Nov follow-up (Claude): straightening experiment (notebook 14, section 9).** The rules were frozen before reading (`claude_margin_diagnostic_20261109/straighten/PROTOCOL_straighten.json`). The 125 long lines were compared in three versions:
- T: the existing tightened crops;
- R: straightened by an image-only deskew, then re-tightened with the same rule (every stroke kept, all crops visually checked);
- H: T stretched horizontally to R's width at 64 px.

Same six frozen readers and static CPU scorer; the T re-read reproduced every saved reading.
- **Supported.** On the 40 sloped lines, S1avg CER fell from 24.9% to 7.9% (−16.9 points, CI −20.8 … −12.2), with 33 better and 4 worse. Deletions fell from 202 to 70, and all six readers improved.
- **Flat controls unharmed:** −0.5 points (CI −2.0 … +0.1).
- **Mechanism.** Stretching without rotating does as well (7.4%, R − H +0.5, CI −0.3 … +1.4). The cause is horizontal resolution: a tall box leaves the text covering too few of S1's 128 positions.
- **Persistent crops:** 26 → 12. The rest are 6 words, 4 sloped lines whose boxes stay tall from residue, and 2 DS lines with a kept blank band.
- **Regressions:** 31, all small (max +7.1 points).
- These are dataset crops, not the app path. No training or app change.

### 9 November — Codex completed JH full-photo app analysis and notebook 16

The user ran `run_jh_full_photos_app.py` on the Mac; its saved log and run identity confirm completion. Codex completed the existing analysis with portable Mac/Linux font and path handling, then built and executed [notebook 16](whiteboard-ocr/notebooks/16_jh_full_photo_app_evaluation.ipynb). All seven code cells executed without errors. No further OCR inference or training was run, and app code remained unchanged. [Full report](claude_jh_fullphoto_20261109/REPORT.md); [all-line review gallery](claude_jh_fullphoto_20261109/review.html); independent checks in `claude_jh_fullphoto_20261109/CHECKS_CODEX.json`.

The seven JH full photos were processed by the unchanged mobile detector, grouping, quad crop and S1 seed-1 reader. All 59 supplied dataset lines have one-to-one geometric associations after grouping, so none was excluded from the matched recognition comparison. Mean line CER for those same 59 lines: original dataset crops **21.14%**, tightened **16.43%**, straightened **5.37%**, actual app crops **0.87%**. Deleted characters: **232 / 187 / 68 / 5** respectively. S1 read 46/59 app lines exactly after case/whitespace normalisation; none reached 15% CER. App corpus CER is 20 edits / 2,355 reference characters = **0.85%**, a different aggregation from mean line CER.

Raw geometry: 58 one-to-one associations, one split line, two unmatched regions. Existing grouping joins the split “Sunlight warms the quiet forest” line correctly, leaving 59 one-to-one associations plus the unmatched regions. Photo 2's unmatched region is the board eraser (empty reading); Photo 5 contains faint “Q Search” writing absent from the supplied inventory (read as “Q 5earch”). *[Corrected 9 Nov, Claude and Codex: this is a projected Windows 11 taskbar (logo, magnifier, “Search”), not writing.]* These are reported separately; they are not silently discarded from a claimed full-page score. No full-page score or reading-order evaluation is claimed.

The actual app crops are already nearly horizontal (median absolute residual slope 0.4 degrees) and wider at input height 64 (median 814 px vs 342 original / 386 tightened / 500 straightened). This supports retaining the current crop path without adding another automatic tightening/deskew stage for these photos. Full photos also contain approximately 1.59x higher-resolution source samples than the dataset crops; geometry, source detail, resampling and compression are confounded. Do not attribute the whole gain solely to horizontal positions or call it broad handwriting generalisation. One inspected development writer, seven photos and provisional assistant-reviewed bands remain the scope.

Independent checks: 236 line/arm CER records recomputed, all 61 app input PNGs matched the saved crop pixels plus exact backend resize, the paired seven-photo intervals reproduced, and photo/source/code/placement hashes matched. Re-read tight/straightened predictions matched their earlier S1 results. All 59 four-way image comparisons and the Photo 2 split/grouping example were visually inspected. The handoff's “at least 160 matches” was inaccurate: saved placements have minima of 110 feature matches, 97 RANSAC inliers and 0.8017 pixel correlation.

Clarifications to the preceding straightening interpretation: the flat-control average passed the stated safety criterion, but individual regressions occurred; similar straightening/stretching scores do not prove equivalence or uniquely identify a mechanism. The model has 128 output positions at width 1024; wider actual inputs can produce more positions. No label-free rule guarantees a position per unknown ground-truth character.

Recommended next milestone: retain and record the provisional mobile-detector/S1 configuration, confirm it on fresh writer/photo groups, and progress deployment rather than starting another backbone experiment based on loose dataset crops. Reading-order work remains deferred by the user. No new training, preprocessing integration or detector change has been selected.

**9 Nov follow-up (Claude): JH full photos through the unchanged app pipeline (notebook 16).** The app path (mobile_det → grouping → order → quad crop → S1) ran on the Mac on `JH_datasetNov5Sentence/Photo1–7` with no labels used. Dataset lines were located by image matching only (SIFT), under rules frozen before any app output was opened.
- **Detection and grouping:** all 59 lines one-to-one after grouping, with none missed, merged or split. One raw split was fixed by grouping. There are 2 extra regions: one blank (the eraser), and the projected Windows 11 taskbar "Search" in Photo 5 (checked on the full photo, and Codex agrees).
- **Recognition (same S1 object, same 59 lines):**

  | crop | mean line CER | missing characters |
  |---|---|---|
  | app | **0.9%** | 5 |
  | dataset original | 21.1% | 232 |
  | dataset tightened | 16.4% | 187 |
  | dataset straightened | 5.4% | 68 |

  - The app's crops are level and wide, a median of 814 px wide at 64 px (tightened: 386). *(Corrected 9 Nov: originally said "short".)*
- **Conclusion:** the poor JH sentence scores in the new-writer benchmark are largely associated with the benchmark crops; geometry, source resolution, resampling and compression are confounded (see Codex entry above), so this is not proof of broad generalisation. Re-check training comparisons on corrected or app-generated crops. *(Wording softened 9 Nov.)*

Evidence: `claude_jh_fullphoto_20261109/` and `whiteboard-ocr/outputs/jh_full_photos_app/`. No training or app change.

### 9 November — Claude: inference package, local verification, deployment prep, fresh-photo kit

**What:** froze the provisional configuration (mobile_det → grouping/order → quad crop → S1 seed 1) as a self-contained inference package: [`board2notes_ocr_package/`](board2notes_ocr_package/README.md), about 107 MB. No training, model experiment, detector change, tightening, deskew or reading-order work. The live app and notebooks are unchanged.

- **Contents:**
  - detector and S1 weights;
  - vendored app code with 6 listed patches: package-relative model paths, no download, detector forced to plain CPU Paddle without oneDNN, optional torch, packaged sample photo;
  - pinned requirements, `MANIFEST.json` with settings, dictionary hash and SHA-256 of all 22 model/code/fixture files;
  - 4 regression photos.
  - Excluded: checkpoints, datasets, notebooks, credentials and keys.
- **Verified on a Linux aarch64 VM** (Paddle 3.2.2, because 3.3.1 has no aarch64 wheel; the detector needs 1 thread there): see [VERIFICATION.md](board2notes_ocr_package/VERIFICATION.md).
  - Integrity check passes and the tamper test is detected.
  - **All 4 fixture photos are identical to the saved Mac app outputs** (63/63 texts, boxes within 6e-5 px).
  - PNG, HEIC and EXIF-rotated inputs are read; unreadable and truncated files are rejected with the app message; blank and tiny images give no lines.
  - Export round-trips, the 21 app tests and the real-model end-to-end UI check pass; the server health endpoint answers.
  - These are implementation checks, not accuracy.
- **Benchmark (VM, 4 vCPU):** cold first request 5.5 s; warm 2.8–15 s per photo; requests serialised by the service lock; peak 1.85 GB.
- **Deployment prepared, not provisioned:** linux/amd64 Dockerfile with packaged weights and a build-time integrity check, plus [DEPLOYMENT.md](board2notes_ocr_package/deploy/DEPLOYMENT.md). Host and cost need a team decision.
- **Fresh-photo confirmation kit**, not started: [PROTOCOL.md](board2notes_ocr_package/fresh_photo_confirmation/PROTOCOL.md).
  - 10–15 photos from 2–3 new writers (no team members, ZB, JH, DS or Jy).
  - Two-person transcriptions; rules fixed and hash-frozen before predictions; single run.
  - Main metric is page CER in app order (as notebook 15); the reading-order cost and region diagnostics are reported separately.
  - Tooling was exercised on throwaway fake data only.
- **Incidental observation:** re-saving the dense Jy 8 photo as JPEG q90 changed its reading order and several word readings (36 page edits / 342 characters). It is one observation, not a robustness result, and it is why originals are required.
- **Open:**
  - Mac packaged-vs-live run (`--against-app`) and the x86 container run;
  - browser copy/download check;
  - deployment destination and cost;
  - fresh photos;
  - the duplicate notebook 16 files (keep Codex's `_evaluation` as canonical).
- **Correction:** the Photo 5 extra region is the projected Windows 11 taskbar, confirmed on the full photo (`claude_jh_fullphoto_20261109/q_search_check/`), and Codex agrees. The "Q Search" writing wording remains in `claude_jh_fullphoto_20261109/REPORT.md`, `RESULTS.json`, `review.html` and `16_jh_full_photo_app_evaluation.ipynb`. Also corrected above: "short" crops → "wide", and the overstated conclusion was softened.

Status summary: [STATUS.md](board2notes_ocr_package/STATUS.md).

### 9 November — Claude: fixes after Codex's package review

Codex's five review points were checked against the files, and all were correct.

1. **Container permissions.** All package files were owner-only (600 files, 700 folders). Docker keeps those modes and makes root the owner, so uid 10001 could not read them.
   - Fixed: a first build stage now sets files to 644 and folders to 755, and the final image copies that result, so the weights layer is not duplicated.
   - A build step running as uid 10001 now fails if any file is unreadable, and re-runs the integrity check.
   - The normalisation command was tested on an owner-only copy. **The Docker build itself is still untested**, because Docker is not available on the VM.
2. **Recording a reference in the container** wrote to the read-only `fixtures/expected/`. Fixed: `verify_package.py --record-dir outputs/expected` (tested), `--reference` also looks there, and DEPLOYMENT.md was updated.
3. **Unexpected-file detection covered only `models/`.** Fixed: it now covers `models/`, `b2n_ocr/` and `fixtures/photos/`, ignoring Python caches and `.DS_Store`. Tested: extra files in code and in photos are reported with exit 1. Docs now say which folders are hash-checked.
4. **Photo 5 is a projected Windows 11 taskbar** (corrections above).
5. **0.87 % is not the only accuracy evidence.** The package docs and MANIFEST now cite notebook 15 (27.5 % page CER on 16 Jy photos for this exact configuration) and earlier evaluations. Reading order is described as provisionally evaluated and diagnosed, with improvement deferred.

The integrity check still passes. Model files and vendored code are unchanged, and the 22 manifest hashes are the same. Next: verify a fresh Mac environment against the live app, build and test the x86 container, then choose the host and collect fresh photos. No training.

### 9 November — Codex: checked package fixes and corrected taskbar context

Codex inspected the updated Dockerfile (permission-normalising source stage and non-root verification), verifier (`--record-dir outputs/expected`, reference lookup and unexpected-file checks in models/code/fixture-photo folders) and documentation. The current 22-file integrity check passes on the existing Mac environment. This is code/integrity verification, not an actual Docker build or fresh-environment Mac test; those remain pending.

The taskbar correction is now also applied to Codex's `claude_jh_fullphoto_20261109/REPORT.md`, `review.html`, generated `RESULTS.json` visual-context annotations and notebook `16_jh_full_photo_app_evaluation.ipynb`. The wider context shows a projected Windows taskbar, not handwritten “Q Search”. The saved OCR text remains `Q 5earch`; it was not corrected to hide a model output. Regeneration sources were updated, and all seven notebook code cells executed successfully.

After removing only the newly added context annotations, the regenerated results reproduce the earlier independently verified SHA-256 `470ff513e2b201c8bf462a16404d97b256fde7435156b0f57d9f523635fc6f76`. Scores, geometry associations and per-line saved readings are therefore unchanged. Duplicate notebook 16 files and the original application were left in place. No inference, training, dependency installation or deployment was performed in this correction.

### 9 November — Codex: organised the research notebooks for teammates

Created [board2notes_notebooks/](board2notes_notebooks/README.md), with an executed [project overview](board2notes_notebooks/00_project_overview.ipynb), four main evidence notebooks (10, 14, 15, 16), three supporting notebooks (11–13), and twelve earlier-work notebooks. The detailed entries are relative symlinks to the canonical sources, so existing generators and links remain valid and future notebook edits do not create conflicting copies. Alternate notebook 16 and backup files remain untouched at their original locations.

The overview explains the professor's feedback, data/pooling/training-strategy comparisons, crop-scale diagnosis, actual detector/cropping path, evaluation limitations and pending deployment checks. It reads existing results and images only. All six code cells executed successfully; all nineteen links resolve, and all nineteen source hashes remained unchanged. The historical `09_reader_comparison.ipynb` has a strict-schema metadata issue, recorded in the index without rewriting the source.

The organised folder is a local navigation guide, not a self-contained model/research archive. Rerunning the evidence notebooks still needs the existing result/data folders and research environment. No model inference, training, app changes or deployment was performed. Rebuild script: `whiteboard-ocr/scripts/organise_project_notebooks.py`.

### 9 November — Codex: added resize-height visualizer

Added [notebook 17](board2notes_notebooks/02_supporting_evidence/17_resize_height_visualizer.ipynb), which displays the same JH long-line crop at 32, 48 and 64 pixels while preserving its aspect ratio. It reports the resized width and approximate useful horizontal slots (`width / 8`) so the height choice can be explained visually. The notebook executed successfully with one image chart, no OCR, no training and no model changes. It is included in the organised notebook index as supporting evidence.

### 9 November — Claude: mobile recogniser reconciliation (no GPU training)

The teammate's mobile files and Codex's `UPDATE_FOR_CLAUDE_MOBILE_20261109.md` were checked against the source and by scoring on CPU. Full record: [board2notes_mobile_20261109/RECONCILIATION.md](board2notes_mobile_20261109/RECONCILIATION.md). Server experiments, packages and results are unchanged.

**Findings**

- **T0 as received is a fused (re-parameterised) inference export with no NRTR head.** It loads exactly into a fused rebuild (maximum difference 0.0). It can be scored, but it cannot be the shared training start. The training checkpoint, dictionary file, label lists, log and Colab source were not supplied, so its training-data history is unknown.
- **T0 scores, mean line CER:**

  | set | T0 | server S0 |
  |---|---|---|
  | whiteboard val | 16.1 % | 7.5 % |
  | CNN test | 32.5 % | 25.5 % |
  | new writers | 39.6 % | 26.8 % |
  | held-out long sentences (squeezed to 640 / not squeezed) | 41.7 % / 11.9 % | 2.4 % |

  It shows no sign of memorising our evaluation crops, but that is not proof. Its repeated-letter insertions are consistent with training at 40 positions and reading at 80.
- **Dictionary:** custom 94-character ASCII set. The YAML's dictionary path is a 63-character file in standard PaddleOCR. 4 validation labels contain θ or → that T0 cannot output, and 12 training originals contain ≈, × or η.
- **Pooling:** unpatched LCNet trains at 40 positions and reads at width/8, using 3 of 4 feature rows at 64 px. LCNet patches were added.
- **Loss weights:** the nested YAML weights are ignored, so the effective weights are 1.0/1.0. NRTR was not disabled by that YAML.
- **Guidance:** the gradient check confirms Codex's description.
- **SE:** no YAML interface existed. An explicit SE was added to `blocks6.3` with near-identity initialisation (gate 0.99), and it was verified to train.
- **Stage-wise LR:** LCNet has 6 groups; mapped as `[0.1, 0.1, 0.1, 0.25, 0.5, 1.0]`.

**Plan changes**

- Start from the official mobile pretrained weights now (MC0's lists are byte-identical to server S0's). Alternatively, start from T0's **training** checkpoint after `tools/check_t0_provenance.py` passes.
- 64×1024 with width/8 positions for all arms; explicit loss weights; server budget and schedule unchanged.
- Mobile pooling pair on clean S0 data (1,052 lines).
- **10 GPU runs:** MC0, MG, ML and ME with 2 seeds each, plus MP40/MP128. Estimated about 3 GPU-hours, to be re-measured after the RunPod smoke test.

**Package**

`board2notes_mobile_20261109/runpod/` (`b2n_mobile.py` on top of the unchanged server `b2n_versions.py`). Its smoke test passed end to end on CPU with stand-in weights (`reports/cpu_plumbing_smoke/`). Not launched; it awaits the start decision and approval.

### 9 November, late — Claude: T0's training data identified (teammate's Colab)

The pinned Colab (`PaddleOCR_Fine-Tune.ipynb`, owner 02yuanweic2) trains PP-OCRv5_mobile_rec from the official weights on Hugging Face `TYW02/whiteboard-notes-v1` (rev `3c639f24`). That dataset holds 8,356 training and 2,090 test lines, every original with 7 augmented copies, and no long sentences.

**Overlap by file name and label** ([findings](board2notes_mobile_20261109/T0_teammate/DATA_FINDINGS.md)):
- 133 of 136 whiteboard-validation crops;
- 174 of 174 CNN-test crops;
- all 2,384 Jy files;
- all 70 DS and 31 JH short-word new-writer crops;
- none of the JH sentences, the held-out long sentences or IAM.

**Consequences:**
- Any T0 checkpoint has seen our checkpoint-selection set and most test sets. The mobile experiments will therefore start from the **official pretrained weights** with our unchanged S0 lists. That also follows the teammate's "keep the dataset the same".
- T0 scores on those sets must be labelled "seen in training".
- The Colab log confirms that the NRTR weight 0.0 is ignored.
- The current Colab run (64×960, 40-position pooling, 1,039 labels over 40 characters) is struggling at about 61 % CER on its own leaky test list.


- **Decision (ZB, 10 Nov 00:59):** use our own dataset and splits for all experiments; do not adopt the teammate's HF split. Its file-level random split puts 2,051 of 2,090 test files (1,311 crops) in training as augmented siblings (`board2notes_mobile_20261109/T0_teammate/teammate_split_leaking_crops.csv`).
- **Codex review of the 17-run package (10 Nov), checked by Claude.**
  - (1) MT budget: confirmed. Mixed heights give batch sizes 16/10/8, so 378 updates/epoch and 8,316 updates at 22 epochs. Fixed to 16 epochs = 6,048 updates, from the sampler length.
  - (2) Mobile verdicts missing from the shared report: confirmed. Added a mobile primary rule, 2x2 interaction and repetition sections.
  - (3) RecAug keys ignored: confirmed and documented. The RecConAug limit had already been set to 100.
  - (4) The data batch includes the 64 uncertain-writer crops: confirmed; caveat added.
  - Also: the mobile recipe now uses 64x960 (ZB decision). The RunPod smoke test is still required before training.

- CPU plumbing smoke v3 passed (10 Nov 03:35, stand-in weights) after the Codex fixes and the 960 px change: all 8 setups train, score and export (30/30 identical); `reports/cpu_plumbing_smoke_v3/`. RunPod smoke still pending.
- RunPod smoke attempt 1 (10 Nov, RTX 4090): setup, weights (sha 04745475...) and data checks passed; stopped at the MT-vs-MP40 config check (epoch_num differs by design after the budget fix; CPU test budgets hid it). Check fixed to allow epochs and to require matched updates; verified at full budget. Upload board2notes_mobile_runpod_20261110_v2.zip.
- RunPod smoke attempt 2: real-weight structure checks passed on RTX 4090; smoke_MC0 trained 24 updates, but the export check failed (80% identical text, mean CER diff 0.0046). Cause: export fuses LCNet branches (rep()), small numeric differences flip near-tie characters in an under-trained model (CPU: max |dp| 0.003, 100% frame agreement). Added a frame-level fallback (>= 99% of output frames agree, plus the CER tolerance), recorded in verified.json. Upload board2notes_mobile_runpod_20261110_v3.zip.
- RunPod export diagnostic (10 Nov 04:30, smoke_MG): checkpoint vs export on CPU agree (max |dp| 0.0006, 100% text); the un-fused checkpoint read on GPU is the outlier (max |dp| 0.93, 67-70% text, CER 0.149 vs 0.131-0.137). Decision: all scoring and export checks on CPU (deployment-equivalent); training stays on GPU. Upload board2notes_mobile_runpod_20261110_v4.zip.

## 2026-11-10 16:30 SGT: teammate's HF dataset, leakage fix for V2his / V3his
- Downloaded `TYW02/whiteboard-notes-v1` @ `896d42ec` (matches the lists he sent) into `whiteboard-notes-v1_896d42ec/`.
- His new crop-level split fixed augmented copies crossing train/test (6 JH crops left), but his whiteboard images are repeat photos of the same boards (e.g. lecture_11/lecture_12 share 32 of 38 phrases): 298 of 390 test crops had the same writing in train.
- Fix (agreed with the user; teammate agreed to group by label): each piece of writing is one group, wholly in train or test. `board2notes_mobile_20261109/his_data/prep_his_data.py`, seeded; reproduced identically here and in the cloud workspace.
- Result: train 1,499 crops (8,232 files for V3his, 1,496 originals for V2his), test 409 originals; 352 crops changed side; pixel check: no test/train pair >= 0.9 similarity.
- Our own split checked the same way: 0 val_wb and 0 test_cnn crops share text with our train.
- Batch A at 15:25: 11 of 17 runs done, finish expected ~20:30.
- 16:20: fixed copy written in his format: `whiteboard-notes-v1_896d42ec_fixed/` (images physically moved, rec_gt_train.txt 8,232 / rec_gt_test.txt 2,214 lines, moves.csv). Verified: no missing image, no label changed, images byte-identical, no crop or same-source label on both sides. Original download left unchanged. Script now 283197e3 (adds --fixed-copy; lists unchanged).
- 16:45: second leakage review of the fixed copy. Found one more same-writing group ("so had." / "so hard.", same quiz board, one word corrected): 5-letter near-identical labels are now grouped too. Regenerated `whiteboard-notes-v1_896d42ec_fixed/` (first version moved to `_to_delete/`): train 1,503 crops (8,252 files; 1,500 originals), test 405; 376 crops moved. Review: no byte-identical file, crop name or same-source label across the split; pixel check of 405 test originals vs all 8,252 train files max 0.836 (different words). Script 19fc02f7.
- 17:10: teammate re-uploaded HF dataset `75619db2` (16:20 SGT), regrouped by label: same images and labels, rearranged. Reviewed: ~60 crops still had the same writing on both sides (punctuation variants, edge-cut crops, contained lines, one mislabelled trio). Script (d416c30f) now defaults to 75619db2 with fewest-moves assignment plus a pixel link (>= 0.85, same source): 64 crops moved; train 1,511 / test 397 crops; review clean (pixel max 0.802). Label error: lecture_05_035 / 06_039 / 07_035 are the same writing with three labels ('4. ResNet', '4.3 Leaky ReLU', '4.2 ReLU'); also in our own train.
- 18:30: agreed plan for his data (75619db2): every group of the same writing that his split had on both sides of train/test moves wholly into a NEW validation set; everything else keeps his side. Reviewed by eye: 7 candidate groups were different words (Chapter N, Classification/Convolution/Kernel headings, Jy similar words) and are excluded. Result: train 1,431 crops (7,887 files; 1,428 originals), val 174, test 303. Three-way checks clean (no shared bytes, crops, same-source or near labels; pixel max 0.80 test-train, 0.72 val-train, 0.67 test-val). Written to `whiteboard-notes-v1_75619db2_fixed/` (train/ val/ test/, rec_gt_train/val/test.txt). Script 02726af2. Old 896d42ec folders moved to `_old_versions/`; the interrupted partial fixed folder to `_to_delete/`.
- 18:45: user chose a stratified 80:10:10 split for his data (option A). Whole groups of the same writing split within every source x line-length band, seeded; ties keep his side. Result train 1,560 / val 179 / test 169 crops (81.8 / 9.4 / 8.9 %); long lines 137 / 19 / 17. 621 crops differ from his split. All three pairs leak-checked clean (pixel max 0.72). `whiteboard-notes-v1_75619db2_fixed/` regenerated (train/ val/ test/); the 75:9:16 version moved to `_old_versions/`. Script 860f461d. Lists for batch B: his_train_full (V3his 8,508 files), his_train_orig (V2his 1,557), his_val_orig (179, checkpoint selection), his_test_orig (169).
- 19:35: batch B package built: `board2notes_mobile_runpod_20261110b.zip` (b2n_mobile.py 3861b841; b2n_versions.py unchanged a4f4ff15). Runs: B0 (teammate's 10 Nov YAML at 640 on MT's lines), MGWlo/MGWlos2/MGWhi/MGWhis2 (NRTR 0.5 / 2.0 with guidance OFF; user chose this after noting that with guidance on each part gets one loss and Adam makes the weight near-irrelevant), V2his/V3his (his stratified split, best checkpoint on his val, scored on his test only). Reuses batch A's scored controls with hash checks (--reuse-from mobile_official_20261110). Offline checks passed: config diffs, teammate lists (V2his 1,554 / V3his 8,493 / val 179 / test 169 after the shared 98-character limit), report sections. Smoke on the pod runs first.
- 19:50: user: the full working model should be the SERVER reader; B0 dropped (it would duplicate batch A's ML). Added `b2n_server_his.py` + `run_server_his.sh` to the batch B zip: server S1 recipe on his leak-fixed 80:10:10 train (S1his, 8,493 files incl. his copies; S1his_orig optional), best checkpoint on his validation, scored on his test, exported. Chain: batch A -> batch B mobile (6 runs) -> S1his -> pod stops on success.
- 22:00: batch A (mobile, 17 runs) read from run_mobile.log (results zip still on the pod). Mobile control MC0avg: val 9.91 %, CNN 30.6 %, new writers 32.3 %, IAM 25.3 %, held-out long 3.7 % (server S0: 7.52 / 25.5 / 26.8 / 23.9 / 2.4): mobile is ~2.4 points worse on validation and ~5 on CNN/new writers. Primary rule (val, 2 seeds): MG guidance off -0.55 (-1.16..+0.08), both seeds better, not established; ML stage-wise LR +0.21, not established, and worse on CNN in both seeds (+1.5) - unlike server S1; ME extra SE +0.76 (+0.05..+1.66), both seeds worse: hurts. Seed noise MC0s2-MC0 = -0.85 on val. Data: same pattern as server (own data >> IAM; long lines help CNN -3.8 and new writers -3.4; pre-made copies help short-only but hurt with long data). Pooling: MP128 vs MP40 -30 points on val (replicates server). MT (teammate Colab recipe) 44 % val / 70 % CNN, worse than PaddleOCR's original 48x320 recipe by 33 points. T0export worse than MC0 everywhere.

- 11 Nov 18:05: C0 package prepared (`board2notes_c0_20261111.zip`, unzip in /workspace; runs `run_server_c0.sh`, experiment server_c0_20261111). C0 = V2's crops without augmented copies, PaddleOCR's original recipe (48x320, 40 positions), so the server ladder changes one thing per step: pretrained -> C0 (fine-tune on our crops) -> C1 (+ augmented copies) -> P40 (64x1024 input, same 40 positions) -> P128 (width/8 pooling) -> V3 (+ lines needing more than 40 positions). Checked locally on the data zip: every list byte-identical to versions_2x2 (train_C1 = train_P40 = train_P128); C0 = 1,113 lines = exactly C1's 4,269 lines minus 3,156 augmented copies, each copy's crop in C0. On the pod the script also stops unless the pretrained reader reads whiteboard validation 100 % identically to versions_2x2 (same scorer). Not trained yet.
- 11 Nov 18:55: C0 done (results_server_c0_20261111/, zip test OK; lists identical to versions_2x2, pretrained reads whiteboard val 100 % identically, export 100 % identical). Whiteboard val / CNN / held-out long (mean line CER %): pretrained 15.48 / 45.79 / 30.93; C0 8.88 / 29.00 / 7.84; C1 9.30 / 28.72 / 8.81; P40 54.54 / 93.88 / 45.26; P128 6.44 / 26.60 / 4.55; V3 7.18 / 24.27 / 2.65. Steps: fine-tuning C0 vs pretrained -6.60 / -16.79 / -23.09 (IAM +18.4, general handwriting forgotten); augmented copies C1 vs C0 +0.42 [-0.49, +1.46] / -0.28 / +0.97 [+0.04, +2.02] (no gain, slightly worse on long; same pattern as V3 vs V2); wider input alone P40 vs C1 +45.2 (breaks); pooling P128 vs P40 -48.1; together P128 vs C1 -2.86 [-4.87, -1.65] / -2.12 / -4.26 (recalculated from versions_2x2 predictions); long lines V3 vs P128 +0.74 (n.s.) / -2.33 / -1.90. C0 new writers 42.38 % (no GPU ladder for that set; C0 includes the 64 teammate crops). One seed each.
- 11 Nov 19:15: notebook 10 (`whiteboard-ocr/notebooks/10_training_results_visual_explorer.ipynb`, linked from `board2notes_notebooks/01_detailed_evidence/`) updated and re-executed without errors: new 4c (server ladder with C0: rung table, chart, step table with photo-cluster intervals recalculated and checked against the RunPod report, one crop read by every rung), 7c (mobile batch A: same-list server vs mobile, two-seed decisions MG/ML/ME, data and pooling replications, MT), 7d (S1his on the teammate's 80:10:10 split, by length), export checks for C0 and S1his in 8, an 11 Nov update at the top of 9; intro updated. Earlier sections unchanged. Previous version: `_old_versions/10_training_results_visual_explorer_before_20261111.ipynb`; patch script `whiteboard-ocr/scripts/patch_nb10_20261111.py`. Batch B still to add (scores on the pod). Notebooks 14-16 not affected. `board2notes_notebooks/INDEX.json` source hash for notebook 10 is now out of date (refresh with organise_project_notebooks.py on the Mac).
- 12 Nov 00:30: fresh-writer line test built from the 44 photos in `1111_unfamiliar/` (5 boards, 16 writings, 152 crops; 5 lines excluded with reasons): `board2notes_freshwriter_test_20261111/` (manifest, crops, review sheets, scores, tools). Crops: app detector (PP-OCRv5_mobile_det) + ink tightening, every crop checked by eye; labels written-as-is (case-insensitive scoring). Writer identity NOT yet recorded. Server readers on it (mean line CER %, CPU, app runtime): S0 2-seed 12.4, S1 11.5, G0 9.1, V2 5.6, S1his 4.6. G0-S0 -3.3 [-6.0, -0.6] by writing but almost all from board A and seed 1 (seed 2 -1.1, n.s.); seed noise up to 3.7 points. V2-S0 -6.8 and S1his-S1 -6.9: models trained with more writers read these boards much better (unless the writer is in that data). Used for reader comparison, so it is development evidence now, not the frozen fresh-photo confirmation.
- 12 Nov 00:35: batch B (mobile, guidance off, CTC 1.0 with NRTR 0.5 or 2.0, two seeds each) read from `batchB_scores/` against MG/MGs2 (guidance off, 1.0/1.0), same crops checked. Whiteboard val (primary): NRTR 0.5 -0.53 [-1.15, +0.11] (both seeds -0.5, not established); NRTR 2.0 -0.05 (no effect). New writers: NRTR 0.5 +1.76 [+1.01, +2.77] worse in both seeds; NRTR 2.0 +0.89 n.s. IAM: NRTR 0.5 +1.18 worse; NRTR 2.0 -0.11. Reading: less NRTR slightly helps familiar boards but hurts unfamiliar handwriting; more NRTR changes nothing. Keep 1.0/1.0; no server weight runs planned.
- 12 Nov 01:00: new notebook 18 `whiteboard-ocr/notebooks/18_server_reader_progression.ipynb` (linked in `board2notes_notebooks/01_detailed_evidence/`, README and INDEX updated): every server model (pretrained, V1, V1b, C0, C1, P40, P128, V2short, V3short, V3, V2, S0-S3, G0, S1his) on one scorer (pretrained/V1 readings checked identical across experiments, same crops per set), training budgets, the one-change-per-step progression with photo-cluster intervals, two-seed strategy decisions and seed noise, new writers and fresh boards, S1his on the teammate split, and a "what contributed" summary. Executed without errors. Built by `whiteboard-ocr/scripts/make_nb18_server_progression.py`.
- 12 Nov 02:00: weakness diagnosis before any gradient-routing run (no training; files in `board2notes_freshwriter_test_20261111/weakness_diagnosis_20261112/`). New writers: deletions (12.9 per 100 chars) are as large as substitutions (11.0); 19% of crops have fewer CTC positions than the line needs (CER about 50%); widening crops 2x cuts new-writer CER 26.8->9.8 (S0) and 22.7->7.2 (G0). New-writer crops are loosely cut (ink fills 36% of crop height vs 75% on fresh boards). Fresh boards (tight crops, none short of positions): widening hurts (S0 12.4->23.6), so a fixed stretch is wrong; only an adaptive one could help. Guidance off (G0) cuts substitutions 31% but not deletions, so routing can only target the shape-confusion part. Next (proposed, not run): check how tightly the app's own detector cuts lines; per-stage CTC vs NRTR gradients and S0/G0 weight change; NRTR-head readings.
- 12 Nov 14:00: check 1, the app's own line detection on the fresh photos (`weakness_diagnosis_20261112/app_crops/`). Through the real app pipeline, line CER is far worse than on hand-checked tight crops: G0 26.6 vs 9.1, S1his 28.9 vs 4.6. Cause: on large thin-stroke words (boards A-C) the 1600 px quick pass finds letter fragments (median 9 px), the small-text rule re-detects at full resolution, and words break into pieces (35 of 152 lines split, some one-box matches are a single letter). Sentence boards D-E are nearly fine (G0 10.1 vs 7.6). Detecting at 960 px instead: 151 of 152 lines one box, G0 9.6, S1his 5.3. Tuned on this set, so it must be checked on small-writing photos (frozen fresh-photo confirmation, JH full photos) before going into the app. Pod package for checks 2-3 built: `board2notes_routing_diag_20261112.zip` (weights moved per stage, stage-feature CKA, CTC vs NRTR gradients per stage, CTC vs NRTR head readings; no training). Tested end to end on CPU with stand-in weights; not run on the pod yet.
- 12 Nov 15:20: notebook 19 `whiteboard-ocr/notebooks/19_ctc_columns_and_app_ocr.ipynb` (linked in `board2notes_notebooks/01_detailed_evidence/`, README and INDEX updated; built by `whiteboard-ocr/scripts/make_nb19_ctc_columns.py`, executed without errors): the resize step, a real new-writer crop drawn as its CTC columns (44 columns for 59 characters -> words collapse; widened 2x -> 88 columns, read correctly), a wrong-letter example with each column's top-3 guesses (S0 vs G0), the app's real boxes vs labelled lines vs 960 px on IMG_5095 and IMG_5122 with per-line readings, the app's pieces for 'bookkeeper', and the fresh-board summary table.
- 13 Nov 19:55 (wrap-up; no further model development planned):
  - Detector size: lines 32-96 px tall at detection size come out as one whole box 99-100% of the time; below ~16 px they are missed, above 128 px they split into letters. Rule C (quick 960 px pass; if lines are taller than 128 px in the full photo, detect where they are ~64 px, else unchanged) run as real code on 114 photo versions, S1 reader: fresh big words 36.0 -> 12.0% CER, every other set within 0.2, small-writing (zoomed) sets unchanged, mean 10.8 -> 7.7. In a copy of the app file only (`weakness_diagnosis_20261112/additions_20261113/detector_size_sweep/`); the app is NOT changed. Thresholds chosen on these photos; frozen fresh-photo confirmation not collected.
  - Fair guidance test + gradient routing (`results_routing_runs_20261112/`, 8 runs, same 1,359 lines as 7 Nov, checkpoint every 5 epochs, per-seed reading in `ANALYSIS_per_seed_20261113.md`). G0 vs S0 at equal epochs: new writers -2.2 / -0.5 (seed 1 / 2), fresh boards +0.4 / +2.0, so most of G0's earlier gain was training length. Routing CTC to stem + stages 1-2 (RE) lost the new-writer gain (+2.5 / +1.9 vs G0); routing to stages 3-4 (RL) beat RE on new writers (1.2 / 2.0) and fresh boards (2.0 / 2.1) in both seeds, never worse than RE in both seeds at any epoch. RL vs S0: -0.7 new writers, -0.8 fresh boards, +0.25 validation, within seed noise (~2 points), so not adopted. Layer check: S0 and G0 features differ only in stages 3-4 (CKA), CTC is ~99% of the backbone gradient at weights 1:1, G0 moves the stem-stage 3 less than S0 even at equal epochs (corrects the earlier training-length explanation), and CTC's gain on new writers is mostly fewer deletions, not fewer substitutions.
  - Pod package `board2notes_routing_runs_20261112.zip`: fixed during testing (resume setting left in config, GPU float-noise tolerance in the routing test, check now saves its data scan).

- 18 Nov 19:10 (scalability, no training): scalability = extension to handwritten paper notes, reusing the pipeline (rubric: "great scalability, with clear avenues"; Satisfactory band = "requires significant redevelopment").
  - Evidence already measured (versions_2x2, 6 Nov, one run each, checkpoints on whiteboard val): IAM val CER downloaded 12.20, whiteboard reader V2 23.26, paper reader V1b 7.79 (same settings; config diff V1b vs V2 = training list, epoch count to match ~6,000 updates, warmup in epochs, paths). 4,411 of 6,482 IAM train lines (68%) need >40 CTC positions. Mobile pooling pair: MP40 39.59 vs MP128 9.19 whiteboard val.
  - New folder `scalability_paper_notes_20261118/`: writing sheet (3 pages, page 3 Chinese before/after check), reference text, `tools/run_paper_notes_eval.py` (detect once, same crops for downloaded / whiteboard / paper / final / stage-wise readers; reading and end-to-end scores; original vs Board2Slides Automatic preparation). Awaiting the user's photos.
  - Plumbing check (Linux aarch64 VM, Paddle 3.2.0, PaddleOCR 3.7.0): all five readers reproduce recorded predictions (37-40 of 40 identical, mean CER diff 0.00). Hugging Face PP-OCRv5_server_rec (b26c3587) = the experiments' "pretrained", when read at 48 px padded to 1024 px; with PaddleOCR's default resize (min width 320) it reads long lines much worse (14 crops: 16.7% vs 7.0%). On a synthetic ruled page, Board2Slides Automatic preparation cropped to 3 of 11 lines (flat) / 7 of 11 (angled); to confirm on real photos.
  - Found: the Board2Slides app's V2 recogniser is byte-identical to T0 (teammate's mobile export: 94-character ASCII dictionary, trained on our val/CNN-test crops). User: teammate will switch the app to our latest model.
  - `board2notes_reader_progression_20261118.zip` (934 MB, CRC OK): 00 downloaded ... 08 final + two scalability readers, readable folder names, README with scores (`board2notes_reader_progression_20261118_README.md`). First build moved to `_to_delete/progression_zip_first_build_20261118/`.
  - Deck: scalability slides sc1-sc3 added after the summary (sc2 is a placeholder until the photos are scored).
- 18 Nov 23:15 (paper-notes photos scored; no training): ZB wrote one page (both topics, own wording, some lines wrapped; no Chinese page), 6 photos (2 flat, 2 medium angle, 2 strong angle). Reference = transcription of what was written (`reference_text/page1.txt`; the planned text moved to `_planned_text_not_written/`). Matcher fixed (a long crop no longer matches a short line it contains; threshold 65).
  - Flat photo, original, same crops: end-to-end page CER downloaded 3.8, whiteboard reader (V2) 2.6, paper reader (V1b, IAM) 7.7, final (S1his) 3.5; 25 of 25 lines found. Second flat photo 4.2 / 3.6 / 7.8 / 4.1. The IAM-trained reader is the WORST on these print-style phone-photo notes: the IAM gain (12.2 -> 7.8 on IAM) does not transfer; "paper" is not one domain.
  - Angled photos: 9-12 of 25 lines found, ~70% end to end for every reader: the package's line grouping merges tilted lines into one box. Board2Slides Automatic preparation: crop kept 1-4 of 25 lines on every photo; perspective abstained.
  - Open: whether ZB's handwriting is in the whiteboard training data (would favour the whiteboard reader).
