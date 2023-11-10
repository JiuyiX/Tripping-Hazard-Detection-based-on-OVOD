import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict, annotate
import glob
import numpy as np
import argparse

model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

def detect_people(img_path):
    TEXT_PROMPT = "people"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(img_path)
    h, w, _ = image_source.shape

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    if len(boxes) > 1:
        logits_arr = np.array(logits)
        maxindex = np.argmax(logits_arr)
        boxes_arr = boxes[maxindex]
        boxes = boxes_arr.reshape((1, 4))
        logits_arr = logits_arr[maxindex].reshape(1)
        logits = torch.tensor(logits_arr)
        phrases = ['people']

    # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    # cv2.imwrite(img_path, annotated_frame)

    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    width = xyxy[0][2] - xyxy[0][0]
    height = xyxy[0][3] - xyxy[0][1]

    h_threshold1 = xyxy[0][3]
    h_threshold2 = xyxy[0][3] - height * 0.5

    w_threshold1 = xyxy[0][0] - width * 0.1
    w_threshold2 = xyxy[0][2] + width * 0.1
    w_threshold3 = xyxy[0][0] + width * 0.25
    w_threshold4 = xyxy[0][2] - width * 0.25

    return h_threshold1, h_threshold2, w_threshold1, w_threshold2, w_threshold3, w_threshold4

def detect_hazard(img_hazard_path):
    TEXT_PROMPT = "electric wire.plastic sheet.brick.roadblock.carton.cardboard.trash pile.wooden slot.bottle.tire.ladder.hole.metal bar.metal rack"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    TN = 0
    FN = 0

    for img_path in glob.glob(img_hazard_path):
        print(img_path)
        h_th1, h_th2, w_th1, w_th2, w_th3, w_th4 = detect_people(img_path)
        image_source, image = load_image(img_path)
        h, w, _ = image_source.shape

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite(img_path, annotated_frame)

        flag = 0
        if len(boxes) > 0:
            boxes_size = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes_size, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            print(xyxy)
            print(phrases)

            for box_size in xyxy:
                if (box_size[3] > h_th1 and box_size[1] > h_th2) and (not (box_size[0] < w_th1 and box_size[2] > w_th2)) \
                        and (box_size[2] > w_th3 and box_size[0] < w_th4):
                    flag = 1
                    break

            if flag == 1:
                TN += 1
            else:
                FN += 1

        else:
            FN += 1

    return TN, FN

def detect_nohazard(img_nohazard_path):
    TEXT_PROMPT = "electric wire.plastic sheet.brick.roadblock.carton.cardboard.trash pile.wooden slot.bottle.tire.ladder.hole.metal bar.metal rack"
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    TP = 0
    FP = 0

    for img_path in glob.glob(img_nohazard_path):
        print(img_path)
        h_th1, h_th2, w_th1, w_th2, w_th3, w_th4 = detect_people(img_path)
        image_source, image = load_image(img_path)
        h, w, _ = image_source.shape

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        # cv2.imwrite(img_path, annotated_frame)

        flag = 0
        if len(boxes) > 0:
            boxes_size = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes_size, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            print(xyxy)
            print(phrases)

            for box_size in xyxy:
                if (box_size[3] > h_th1 and box_size[1] > h_th2) and (not (box_size[0] < w_th1 and box_size[2] > w_th2)) \
                        and (box_size[2] > w_th3 and box_size[0] < w_th4):
                    flag = 1
                    break

            if flag == 1:
                FP += 1
            else:
                TP += 1

        else:
            TP += 1

    return TP, FP

def detect_evaluation(test_set):
    if test_set == "source_scene":
        img_hazard_path1 = r"TH_scene_source/hazard/scene1/*/*.jpg"
        img_nohazard_path1 = r"TH_scene_source/nohazard/scene1/*/*.jpg"

        img_hazard_path2 = r"TH_scene_source/hazard/scene2/*/*.jpg"
        img_nohazard_path2 = r"TH_scene_source/nohazard/scene2/*/*.jpg"

        img_hazard_path3 = r"TH_scene_source/hazard/scene3/*/*.jpg"
        img_nohazard_path3 = r"TH_scene_source/nohazard/scene3/*/*.jpg"

        img_hazard_path4 = r"TH_scene_source/hazard/scene4/*/*.jpg"
        img_nohazard_path4 = r"TH_scene_source/nohazard/scene4/*/*.jpg"

        img_hazard_path5 = r"TH_scene_source/hazard/scene5/*/*.jpg"
        img_nohazard_path5 = r"TH_scene_source/nohazard/scene5/*/*.jpg"

        img_hazard_path6 = r"TH_scene_source/hazard/scene6/*/*.jpg"
        img_nohazard_path6 = r"TH_scene_source/nohazard/scene6/*/*.jpg"

        img_hazard_path7 = r"TH_scene_source/hazard/scene7/*/*.jpg"
        img_nohazard_path7 = r"TH_scene_source/nohazard/scene7/*/*.jpg"

    elif test_set == "correction_scene":
        img_hazard_path1 = r"TH_scene_correction/hazard/scene1/*/*.jpg"
        img_nohazard_path1 = r"TH_scene_correction/nohazard/scene1/*/*.jpg"

        img_hazard_path2 = r"TH_scene_correction/hazard/scene2/*/*.jpg"
        img_nohazard_path2 = r"TH_scene_correction/nohazard/scene2/*/*.jpg"

        img_hazard_path3 = r"TH_scene_correction/hazard/scene3/*/*.jpg"
        img_nohazard_path3 = r"TH_scene_correction/nohazard/scene3/*/*.jpg"

        img_hazard_path4 = r"TH_scene_correction/hazard/scene4/*/*.jpg"
        img_nohazard_path4 = r"TH_scene_correction/nohazard/scene4/*/*.jpg"

        img_hazard_path5 = r"TH_scene_correction/hazard/scene5/*/*.jpg"
        img_nohazard_path5 = r"TH_scene_correction/nohazard/scene5/*/*.jpg"

        img_hazard_path6 = r"TH_scene_correction/hazard/scene6/*/*.jpg"
        img_nohazard_path6 = r"TH_scene_correction/nohazard/scene6/*/*.jpg"

        img_hazard_path7 = r"TH_scene_correction/hazard/scene7/*/*.jpg"
        img_nohazard_path7 = r"TH_scene_correction/nohazard/scene7/*/*.jpg"

    elif test_set == "source_indoor":
        img_hazard_path1 = r"TH_in_outdoor_source/indoor/hazard/scene1/*.jpg"
        img_nohazard_path1 = r"TH_in_outdoor_source/indoor/nohazard/scene1/*.jpg"

        img_hazard_path2 = r"TH_in_outdoor_source/indoor/hazard/scene2/*.jpg"
        img_nohazard_path2 = r"TH_in_outdoor_source/indoor/nohazard/scene2/*.jpg"

        img_hazard_path3 = r"TH_in_outdoor_source/indoor/hazard/scene3/*.jpg"
        img_nohazard_path3 = r"TH_in_outdoor_source/indoor/nohazard/scene3/*.jpg"

        img_hazard_path4 = r"TH_in_outdoor_source/indoor/hazard/scene4/*.jpg"
        img_nohazard_path4 = r"TH_in_outdoor_source/indoor/nohazard/scene4/*.jpg"

        img_hazard_path5 = r"TH_in_outdoor_source/indoor/hazard/scene5/*.jpg"
        img_nohazard_path5 = r"TH_in_outdoor_source/indoor/nohazard/scene5/*.jpg"

        img_hazard_path6 = r"TH_in_outdoor_source/indoor/hazard/scene6/*.jpg"
        img_nohazard_path6 = r"TH_in_outdoor_source/indoor/nohazard/scene6/*.jpg"

        img_hazard_path7 = r"TH_in_outdoor_source/indoor/hazard/scene7/*.jpg"
        img_nohazard_path7 = r"TH_in_outdoor_source/indoor/nohazard/scene7/*.jpg"

    elif test_set == "correction_indoor":
        img_hazard_path1 = r"TH_in_outdoor_correction/indoor/hazard/scene1/*.jpg"
        img_nohazard_path1 = r"TH_in_outdoor_correction/indoor/nohazard/scene1/*.jpg"

        img_hazard_path2 = r"TH_in_outdoor_correction/indoor/hazard/scene2/*.jpg"
        img_nohazard_path2 = r"TH_in_outdoor_correction/indoor/nohazard/scene2/*.jpg"

        img_hazard_path3 = r"TH_in_outdoor_correction/indoor/hazard/scene3/*.jpg"
        img_nohazard_path3 = r"TH_in_outdoor_correction/indoor/nohazard/scene3/*.jpg"

        img_hazard_path4 = r"TH_in_outdoor_correction/indoor/hazard/scene4/*.jpg"
        img_nohazard_path4 = r"TH_in_outdoor_correction/indoor/nohazard/scene4/*.jpg"

        img_hazard_path5 = r"TH_in_outdoor_correction/indoor/hazard/scene5/*.jpg"
        img_nohazard_path5 = r"TH_in_outdoor_correction/indoor/nohazard/scene5/*.jpg"

        img_hazard_path6 = r"TH_in_outdoor_correction/indoor/hazard/scene6/*.jpg"
        img_nohazard_path6 = r"TH_in_outdoor_correction/indoor/nohazard/scene6/*.jpg"

        img_hazard_path7 = r"TH_in_outdoor_correction/indoor/hazard/scene7/*.jpg"
        img_nohazard_path7 = r"TH_in_outdoor_correction/indoor/nohazard/scene7/*.jpg"

    elif test_set == "source_outdoor":
        img_hazard_path1 = r"TH_in_outdoor_source/outdoor/hazard/scene1/*.jpg"
        img_nohazard_path1 = r"TH_in_outdoor_source/outdoor/nohazard/scene1/*.jpg"

        img_hazard_path2 = r"TH_in_outdoor_source/outdoor/hazard/scene2/*.jpg"
        img_nohazard_path2 = r"TH_in_outdoor_source/outdoor/nohazard/scene2/*.jpg"

        img_hazard_path3 = r"TH_in_outdoor_source/outdoor/hazard/scene3/*.jpg"
        img_nohazard_path3 = r"TH_in_outdoor_source/outdoor/nohazard/scene3/*.jpg"

        img_hazard_path4 = r"TH_in_outdoor_source/outdoor/hazard/scene4/*.jpg"
        img_nohazard_path4 = r"TH_in_outdoor_source/outdoor/nohazard/scene4/*.jpg"

        img_hazard_path5 = r"TH_in_outdoor_source/outdoor/hazard/scene5/*.jpg"
        img_nohazard_path5 = r"TH_in_outdoor_source/outdoor/nohazard/scene5/*.jpg"

        img_hazard_path6 = r"TH_in_outdoor_source/outdoor/hazard/scene6/*.jpg"
        img_nohazard_path6 = r"TH_in_outdoor_source/outdoor/nohazard/scene6/*.jpg"

        img_hazard_path7 = r"TH_in_outdoor_source/outdoor/hazard/scene7/*.jpg"
        img_nohazard_path7 = r"TH_in_outdoor_source/outdoor/nohazard/scene7/*.jpg"

    elif test_set == "correction_outdoor":
        img_hazard_path1 = r"TH_in_outdoor_correction/outdoor/hazard/scene1/*.jpg"
        img_nohazard_path1 = r"TH_in_outdoor_correction/outdoor/nohazard/scene1/*.jpg"

        img_hazard_path2 = r"TH_in_outdoor_correction/outdoor/hazard/scene2/*.jpg"
        img_nohazard_path2 = r"TH_in_outdoor_correction/outdoor/nohazard/scene2/*.jpg"

        img_hazard_path3 = r"TH_in_outdoor_correction/outdoor/hazard/scene3/*.jpg"
        img_nohazard_path3 = r"TH_in_outdoor_correction/outdoor/nohazard/scene3/*.jpg"

        img_hazard_path4 = r"TH_in_outdoor_correction/outdoor/hazard/scene4/*.jpg"
        img_nohazard_path4 = r"TH_in_outdoor_correction/outdoor/nohazard/scene4/*.jpg"

        img_hazard_path5 = r"TH_in_outdoor_correction/outdoor/hazard/scene5/*.jpg"
        img_nohazard_path5 = r"TH_in_outdoor_correction/outdoor/nohazard/scene5/*.jpg"

        img_hazard_path6 = r"TH_in_outdoor_correction/outdoor/hazard/scene6/*.jpg"
        img_nohazard_path6 = r"TH_in_outdoor_correction/outdoor/nohazard/scene6/*.jpg"

        img_hazard_path7 = r"TH_in_outdoor_correction/outdoor/hazard/scene7/*.jpg"
        img_nohazard_path7 = r"TH_in_outdoor_correction/outdoor/nohazard/scene7/*.jpg"

    else:
        img_hazard_path1 = ""
        img_nohazard_path1 = ""

        img_hazard_path2 = ""
        img_nohazard_path2 = ""

        img_hazard_path3 = ""
        img_nohazard_path3 = ""

        img_hazard_path4 = ""
        img_nohazard_path4 = ""

        img_hazard_path5 = ""
        img_nohazard_path5 = ""

        img_hazard_path6 = ""
        img_nohazard_path6 = ""

        img_hazard_path7 = ""
        img_nohazard_path7 = ""

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    TP1, FP1 = detect_nohazard(img_nohazard_path1)
    TN1, FN1 = detect_hazard(img_hazard_path1)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 1: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    TP1, FP1 = detect_nohazard(img_nohazard_path2)
    TN1, FN1 = detect_hazard(img_hazard_path2)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 2: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    TP1, FP1 = detect_nohazard(img_nohazard_path3)
    TN1, FN1 = detect_hazard(img_hazard_path3)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 3: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    TP1, FP1 = detect_nohazard(img_nohazard_path4)
    TN1, FN1 = detect_hazard(img_hazard_path4)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 4: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    TP1, FP1 = detect_nohazard(img_nohazard_path5)
    TN1, FN1 = detect_hazard(img_hazard_path5)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 5: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    TP1, FP1 = detect_nohazard(img_nohazard_path6)
    TN1, FN1 = detect_hazard(img_hazard_path6)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 6: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    TP1, FP1 = detect_nohazard(img_nohazard_path7)
    TN1, FN1 = detect_hazard(img_hazard_path7)

    if (TP1 + FP1 + TN1 + FN1) > 30:
        TP += TP1
        FP += FP1
        TN += TN1
        FN += FN1

        noh_recall1 = float(float(TP1) / (TP1 + FN1))
        noh_precision1 = float(float(TP1) / (TP1 + FP1))

        h_recall1 = float(float(TN1) / (TN1 + FP1))
        h_precision1 = float(float(TN1) / (TN1 + FN1))

        noh_f1_score1 = float(2 * noh_recall1 * noh_precision1 / (noh_recall1 + noh_precision1))
        h_f1_score1 = float(2 * h_recall1 * h_precision1 / (h_recall1 + h_precision1))

        noh1 = (float(TP1 + FN1) / (TP1 + TN1 + FN1 + FP1))
        h1 = 1 - noh1

        wF1Score1 = noh1 * noh_f1_score1 + h1 * h_f1_score1
        noh_acc1 = (float(TP1) / (TP1 + FN1))
        h_acc1 = (float(TN1) / (TN1 + FP1))

        print("Test on Scene 7: ")
        print("Weighted F1-score is {}".format(wF1Score1))
        print("NoHazard Accuracy is {}".format(noh_acc1))
        print("Hazard Accuracy is {}".format(h_acc1))

    noh_recall = float(float(TP) / (TP + FN))
    noh_precision = float(float(TP) / (TP + FP))

    h_recall = float(float(TN) / (TN + FP))
    h_precision = float(float(TN) / (TN + FN))

    noh_f1_score = float(2 * noh_recall * noh_precision / (noh_recall + noh_precision))
    h_f1_score = float(2 * h_recall * h_precision / (h_recall + h_precision))

    noh = (float(TP + FN) / (TP + TN + FN + FP))
    h = 1 - noh

    wF1Score = noh * noh_f1_score + h * h_f1_score
    noh_acc = (float(TP) / (TP + FN))
    h_acc = (float(TN) / (TN + FP))

    print("Weighted F1-score on the whole dataset is {}".format(wF1Score))
    print("NoHazard Accuracy on the whole dataset is {}".format(noh_acc))
    print("Hazard Accuracy on the whole dataset is {}".format(h_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TH_GroundingDNO')

    parser.add_argument('--IMG_DIR',default='source_indoor',help='The image directory')

    args = parser.parse_args()

    img_dir = args.IMG_DIR
    detect_evaluation(img_dir)