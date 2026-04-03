import os
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

@DATASETS.register_module()
class AirLeishReportDataset(CocoDataset):
    def evaluate(self, results, metric=['bbox', 'segm'], logger=None, work_dir=None, classwise=True, **kwargs):
        # Chiama la valutazione COCO standard
        eval_results = super().evaluate(results, metric, logger, classwise=classwise, **kwargs)
        
        # Se work_dir non è passato, usa la directory corrente
        if work_dir is None:
            work_dir = './'
            
        report_path = os.path.join(work_dir, 'epoch_metrics_report.txt')
        
        # Costruisci il report leggendo dal dizionario eval_results
        report_lines = [
            "="*50,
            "REPORT EPOCHE - MMDET COCO EVALUATION",
            "="*50
        ]
        
        for key, val in eval_results.items():
            # Filtra per salvare solo le metriche che ti interessano e formattarle bene
            if isinstance(val, float):
                report_lines.append(f"{key:<30}: {val:.4f}")
        
        report_lines.append("\n\n")
        
        # Salva su file
        with open(report_path, 'a') as f:
            f.write("\n".join(report_lines))
            
        return eval_results