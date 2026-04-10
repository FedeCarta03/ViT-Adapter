import os
import numpy as np
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from sklearn.metrics import confusion_matrix

@DATASETS.register_module()
class SklearnMetricsDataset(CustomDataset):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Creiamo una matrice vuota che si riempirà durante la validazione
        self.num_classes = len(self.CLASSES)
        self.global_conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def pre_eval(self, preds, indices):
        """Questa funzione intercetta l'immagine cruda prima che venga distrutta dal framework."""
        
        # 1. Aggiorniamo la matrice di Sklearn immagine per immagine
        for pred, idx in zip(preds, indices):
            # Assicuriamoci di estrarre l'array se è una tupla
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
                
            gt = self.get_gt_seg_map_by_idx(idx)
            
            pred_flat = np.array(pred).flatten()
            gt_flat = gt.flatten()
            
            # Rimuoviamo i pixel da ignorare
            valid_mask = gt_flat != self.ignore_index
            pred_flat = pred_flat[valid_mask]
            gt_flat = gt_flat[valid_mask]
            
            # Sklearn calcola la matrice per QUESTA singola immagine
            labels = list(range(self.num_classes))
            cm = confusion_matrix(gt_flat, pred_flat, labels=labels)
            
            # L'aggiungiamo al totale globale
            self.global_conf_mat += cm

        # 2. Lasciamo che mmsegmentation continui il suo lavoro normale
        return super().pre_eval(preds, indices)
        
    def evaluate(self, results, metric='mIoU', logger=None, work_dir=None, **kwargs):
        """Questa viene chiamata alla fine dell'epoca per tirare le somme."""
        
        # 1. Validazione classica nativa
        eval_results = super().evaluate(results, metric, logger, **kwargs)
        
        if work_dir is None:
            work_dir = './'
            
        logger.info("\nGenerazione Report Avanzato Sklearn in corso...")
        
        # 2. Estrazione dalla Matrice di Confusione Globale
        conf_mat = self.global_conf_mat
        
        # Convertiamo subito in float64 per evitare Integer Overflow!
        TP = np.diag(conf_mat).astype(np.float64)
        FP = (conf_mat.sum(axis=0) - np.diag(conf_mat)).astype(np.float64)
        FN = (conf_mat.sum(axis=1) - np.diag(conf_mat)).astype(np.float64)
        TN = (conf_mat.sum() - (FP + FN + TP)).astype(np.float64)
        
        # Support: numero totale di pixel reali per ogni classe (TP + FN)
        support = TP + FN
        total_support = np.sum(support)
        
        # Costante per evitare divisioni per zero
        eps = 1e-7
        
        # --- Calcolo Metriche PER CLASSE ---
        pixel_accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)  # Sensitivity
        specificity = TN / (TN + FP + eps)
        f1_dice = 2 * TP / (2 * TP + FP + FN + eps)
        iou_jaccard = TP / (TP + FP + FN + eps)
        balanced_acc = (recall + specificity) / 2
        
        # Calcolo MCC (Matthews Correlation Coefficient)
        mcc_num = (TP * TN) - (FP * FN)
        mcc_den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = np.divide(mcc_num, mcc_den, out=np.zeros_like(mcc_num, dtype=float), where=mcc_den!=0)

        # --- Calcolo Metriche MICRO AVERAGE ---
        # Sommiamo globalmente TP, FP, FN, TN
        TP_mic = np.sum(TP)
        FP_mic = np.sum(FP)
        FN_mic = np.sum(FN)
        TN_mic = np.sum(TN)
        
        mic_pixel_accuracy = (TP_mic + TN_mic) / (TP_mic + TN_mic + FP_mic + FN_mic + eps)
        mic_precision = TP_mic / (TP_mic + FP_mic + eps)
        mic_recall = TP_mic / (TP_mic + FN_mic + eps)
        mic_specificity = TN_mic / (TN_mic + FP_mic + eps)
        mic_f1_dice = 2 * TP_mic / (2 * TP_mic + FP_mic + FN_mic + eps)
        mic_iou_jaccard = TP_mic / (TP_mic + FP_mic + FN_mic + eps)
        mic_balanced_acc = (mic_recall + mic_specificity) / 2
        
        mic_mcc_num = (TP_mic * TN_mic) - (FP_mic * FN_mic)
        mic_mcc_den = np.sqrt((TP_mic + FP_mic) * (TP_mic + FN_mic) * (TN_mic + FP_mic) * (TN_mic + FN_mic))
        mic_mcc = mic_mcc_num / mic_mcc_den if mic_mcc_den != 0 else 0.0

        # --- Calcolo Metriche WEIGHTED AVERAGE ---
        # Ponderiamo in base al 'support' (pixel effettivi di ground truth per classe)
        if total_support > 0:
            w_pixel_accuracy = np.average(pixel_accuracy, weights=support)
            w_iou_jaccard = np.average(iou_jaccard, weights=support)
            w_f1_dice = np.average(f1_dice, weights=support)
            w_precision = np.average(precision, weights=support)
            w_recall = np.average(recall, weights=support)
            w_specificity = np.average(specificity, weights=support)
            w_balanced_acc = np.average(balanced_acc, weights=support)
            w_mcc = np.average(mcc, weights=support)
        else:
            w_pixel_accuracy = w_iou_jaccard = w_f1_dice = w_precision = w_recall = w_specificity = w_balanced_acc = w_mcc = 0.0

        # 3. Creazione del testo formattato per il Report
        report_lines = [
            "="*60,
            f"REPORT METRICHE AVANZATO - {len(self.CLASSES)} CLASSI",
            "="*60,
            "\n[MATRICE DI CONFUSIONE GLOBALE]",
            f"{conf_mat}\n"
        ]
        
        # Risultati per ogni singola classe
        for i, class_name in enumerate(self.CLASSES):
            report_lines.extend([
                f"--- Dettaglio Classe: {class_name.upper()} ---",
                f"Valori Assoluti (Pixel):",
                f"  Vero Positivo (TP):  {int(TP[i]):>12}",
                f"  Vero Negativo (TN):  {int(TN[i]):>12}",
                f"  Falso Positivo (FP): {int(FP[i]):>12}",
                f"  Falso Negativo (FN): {int(FN[i]):>12}",
                f"  Support (TP+FN):     {int(support[i]):>12}",
                f"\nMetriche:",
                f"  Pixel Accuracy:      {pixel_accuracy[i]:.4f}",
                f"  IoU (Jaccard):       {iou_jaccard[i]:.4f}",
                f"  Dice (F1-Score):     {f1_dice[i]:.4f}",
                f"  Precision:           {precision[i]:.4f}",
                f"  Recall/Sensitivity:  {recall[i]:.4f}",
                f"  Specificity:         {specificity[i]:.4f}",
                f"  Balanced Accuracy:   {balanced_acc[i]:.4f}",
                f"  MCC:                 {mcc[i]:.4f}\n"
            ])
            
        # Aggiungiamo le MEDIE (Macro, Micro, Weighted)
        report_lines.extend([
            "-"*60,
            f"[MEDIE MACRO (Macro-Average)] - Ogni classe conta allo stesso modo",
            f"  Mean Pixel Accuracy: {np.mean(pixel_accuracy):.4f}",
            f"  Mean IoU (mIoU):     {np.mean(iou_jaccard):.4f}",
            f"  Mean Dice (mF1):     {np.mean(f1_dice):.4f}",
            f"  Mean Precision:      {np.mean(precision):.4f}",
            f"  Mean Recall/Sens.:   {np.mean(recall):.4f}",
            f"  Mean Specificity:    {np.mean(specificity):.4f}",
            f"  Mean Balanced Acc:   {np.mean(balanced_acc):.4f}",
            f"  Mean MCC:            {np.mean(mcc):.4f}\n",

            f"[MEDIE MICRO (Micro-Average)] - Aggregazione globale dei pixel",
            f"  Micro Pixel Accuracy:{mic_pixel_accuracy:.4f}",
            f"  Micro IoU:           {mic_iou_jaccard:.4f}",
            f"  Micro Dice (F1):     {mic_f1_dice:.4f}",
            f"  Micro Precision:     {mic_precision:.4f}",
            f"  Micro Recall:        {mic_recall:.4f}",
            f"  Micro Specificity:   {mic_specificity:.4f}",
            f"  Micro Balanced Acc:  {mic_balanced_acc:.4f}",
            f"  Micro MCC:           {mic_mcc:.4f}\n",

            f"[MEDIE WEIGHTED (Weighted-Average)] - Ponderate in base al Support",
            f"  Weight. Pixel Acc:   {w_pixel_accuracy:.4f}",
            f"  Weight. IoU:         {w_iou_jaccard:.4f}",
            f"  Weight. Dice (F1):   {w_f1_dice:.4f}",
            f"  Weight. Precision:   {w_precision:.4f}",
            f"  Weight. Recall:      {w_recall:.4f}",
            f"  Weight. Specificity: {w_specificity:.4f}",
            f"  Weight. Balanced Acc:{w_balanced_acc:.4f}",
            f"  Weight. MCC:         {w_mcc:.4f}",
            "="*60 + "\n\n"
        ])
        
        # Scrittura del Report su file
        report_path = os.path.join(work_dir, 'sklearn_metrics_report.txt')
        with open(report_path, 'a') as f:
            f.write("\n".join(report_lines))
            
        logger.info(f"Report Avanzato salvato con successo in: {report_path}")
        
        # IMPORTANTE: Resettiamo la matrice per la validazione della prossima epoca!
        self.global_conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        
        return eval_results