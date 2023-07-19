# Para-Ref
The official repository of "Not All Metrics Are Guilty: Improving NLG Evaluation with LLM Paraphrasing".

## Requirements
```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .
cd ..

git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..

git clone https://github.com/thompsonb/prism
cd prism
pip install -r requirements.txt
cd ..

pip install -U git+https://github.com/pltrdy/pyrouge
git clone https://github.com/pltrdy/files2rouge.git     
cd files2rouge
python setup_rouge.py
python setup.py install
cd ..

git clone https://github.com/neulab/BARTScore
git clone https://github.com/AIPHES/emnlp19-moverscore

pip install sacrebleu, bert-score, unbabel-comet, pycocoevalcap
```

## WMT22 Metrics Shared Task
Download data:
```bash
alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download
```

Prepare data:
```bash
python data.py zh-en
```

Calculate metric scores for different metrics:
```bash
python score.py score/zh-en.json chrf/bleu/bertscore/bleurt/prism/comet/bartscore/gemba
```

Calculate the correlation scores for different metrics:
```bash
python analysis.py score/zh-en.json chrf/bleu/bertscore/bleurt/prism/comet/bartscore/gemba
```

## SummEval
Download data:
```bash
wget https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl
```

Prepare data:
```bash
python data.py
```

Calculate metric scores for different metrics:
```bash
python score.py score/summeval.json rouge/bertscore/moverscore/chatgpt
```

Calculate the correlation scores for different metrics:
```bash
python analysis.py score/summeval.json rouge/bertscore/moverscore/chatgpt
```

## PASCAL-50S
Download data:
```bash
wget https://filebox.ece.vt.edu/~vrama91/CIDEr_miscellanous/cider_datasets.tar
```

Prepare data:
```bash
python data.py
```

Calculate metric scores for different metrics:
```bash
python score.py score/pascal.json bleu/rouge/meteor/cider/spice/bertscore
```

Calculate the correlation scores for different metrics:
```bash
python analysis.py score/pascal.json bleu/rouge/meteor/cider/spice/bertscore
```