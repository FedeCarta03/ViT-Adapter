import numpy as np

def generate_full_metrics_report(filename="report_metriche.txt"):
    # 1. Definisci la matrice di confusione e i nomi delle classi
    cm = np.array([
        [11412977, 63601, 23],
        [83001, 2721769, 38363],
        [1, 90314, 1842879]
    ], dtype=np.float64)
    
    classes = ["BACKGROUND", "CYTOPLASM", "NUCLEUS"]
    
    # 2. Calcolo dei valori assoluti per classe
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TOTAL = cm.sum()
    TN = TOTAL - (TP + FP + FN)
    
    support = cm.sum(axis=1)
    weights = support / TOTAL
    
    # 3. Calcolo metriche per classe
    pixel_acc = (TP + TN) / TOTAL
    iou = TP / (TP + FP + FN)
    dice = 2 * TP / (2 * TP + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    bal_acc = (recall + specificity) / 2
    
    # MCC per classe
    num_mcc = (TP * TN) - (FP * FN)
    den_mcc = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = num_mcc / den_mcc
    
    # ==========================================
    # PREPARAZIONE DEL TESTO DEL REPORT
    # ==========================================
    lines = []
    lines.append("==================================================")
    lines.append("REPORT METRICHE AVANZATO - COMPLETO")
    lines.append("==================================================\n")
    lines.append("[MATRICE DI CONFUSIONE GLOBALE]")
    lines.append(str(cm.astype(int)) + "\n")
    
    for i, cls_name in enumerate(classes):
        lines.append(f"--- Dettaglio Classe: {cls_name} ---")
        lines.append("Valori Assoluti (Pixel):")
        lines.append(f"  Vero Positivo (TP):    {TP[i]:.1f}")
        lines.append(f"  Vero Negativo (TN):    {TN[i]:.1f}")
        lines.append(f"  Falso Positivo (FP):   {FP[i]:.1f}")
        lines.append(f"  Falso Negativo (FN):   {FN[i]:.1f}\n")
        lines.append("Metriche:")
        lines.append(f"  Pixel Accuracy:      {pixel_acc[i]:.4f}")
        lines.append(f"  IoU (Jaccard):       {iou[i]:.4f}")
        lines.append(f"  Dice (F1-Score):     {dice[i]:.4f}")
        lines.append(f"  Precision:           {precision[i]:.4f}")
        lines.append(f"  Recall/Sensitivity:  {recall[i]:.4f}")
        lines.append(f"  Specificity:         {specificity[i]:.4f}")
        lines.append(f"  Balanced Accuracy:   {bal_acc[i]:.4f}")
        lines.append(f"  MCC:                 {mcc[i]:.4f}\n")
        
    lines.append("--------------------------------------------------")
    lines.append("[MEDIE MACRO (Macro-Average)]")
    lines.append(f"  Mean Pixel Accuracy: {np.mean(pixel_acc):.4f}")
    lines.append(f"  Mean IoU (mIoU):     {np.mean(iou):.4f}")
    lines.append(f"  Mean Dice (mF1):     {np.mean(dice):.4f}")
    lines.append(f"  Mean Precision:      {np.mean(precision):.4f}")
    lines.append(f"  Mean Recall/Sens.:   {np.mean(recall):.4f}")
    lines.append(f"  Mean Specificity:    {np.mean(specificity):.4f}")
    lines.append(f"  Mean Balanced Acc:   {np.mean(bal_acc):.4f}")
    lines.append(f"  Mean MCC:            {np.mean(mcc):.4f}\n")

    tp_sum, fp_sum, fn_sum, tn_sum = TP.sum(), FP.sum(), FN.sum(), TN.sum()
    micro_acc = tp_sum / TOTAL 
    micro_iou = tp_sum / (tp_sum + fp_sum + fn_sum)
    micro_spec = tn_sum / (tn_sum + fp_sum)
    micro_bal_acc = (micro_acc + micro_spec) / 2
    num_mic_mcc = (tp_sum * tn_sum) - (fp_sum * fn_sum)
    den_mic_mcc = np.sqrt((tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum))
    micro_mcc = num_mic_mcc / den_mic_mcc

    lines.append("--------------------------------------------------")
    lines.append("[MEDIE MICRO (Micro-Average)]")
    lines.append("*(Nota: Nei multi-classe esclusivi, Micro Precision, Recall e F1 = Global Pixel Accuracy)*")
    lines.append(f"  Micro Pixel Accuracy: {micro_acc:.4f}")
    lines.append(f"  Micro IoU:            {micro_iou:.4f}")
    lines.append(f"  Micro Dice (mF1):     {micro_acc:.4f}")
    lines.append(f"  Micro Precision:      {micro_acc:.4f}")
    lines.append(f"  Micro Recall/Sens.:   {micro_acc:.4f}")
    lines.append(f"  Micro Specificity:    {micro_spec:.4f}")
    lines.append(f"  Micro Balanced Acc:   {micro_bal_acc:.4f}")
    lines.append(f"  Micro MCC:            {micro_mcc:.4f}\n")

    lines.append("--------------------------------------------------")
    lines.append("[MEDIE PONDERATE (Weighted-Average)]")
    lines.append(f"  Weighted Pixel Acc:   {np.sum(pixel_acc * weights):.4f}")
    lines.append(f"  Weighted IoU:         {np.sum(iou * weights):.4f}")
    lines.append(f"  Weighted Dice (wF1):  {np.sum(dice * weights):.4f}")
    lines.append(f"  Weighted Precision:   {np.sum(precision * weights):.4f}")
    lines.append(f"  Weighted Recall:      {np.sum(recall * weights):.4f}")
    lines.append(f"  Weighted Specificity: {np.sum(specificity * weights):.4f}")
    lines.append(f"  Weighted Bal. Acc:    {np.sum(bal_acc * weights):.4f}")
    lines.append(f"  Weighted MCC:         {np.sum(mcc * weights):.4f}")
    lines.append("==================================================")

    # ==========================================
    # SALVATAGGIO SU FILE E STAMPA A SCHERMO
    # ==========================================
    report_text = "\n".join(lines)
    
    # Stampa a schermo
    print(report_text)
    
    # Scrittura su file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(f"\nOperazione completata! Il report è stato salvato nel file: '{filename}'")

# Esegui lo script
generate_full_metrics_report("report_metriche_5.txt")