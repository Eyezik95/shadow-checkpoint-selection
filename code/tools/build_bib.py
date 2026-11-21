"""Build paper/refs.bib without typing bibliographic data from memory.

* Entries with a DOI are fetched as BibTeX from doi.org (content negotiation, i.e. the publisher's
  Crossref/DataCite record) and only the citation key is replaced.
* timm is written as its README's own citation: its concept DOI resolves to the latest release.
* Entries without a DOI (PMLR, JMLR, ICLR, NeurIPS, MLSys, arXiv, a book) are written from the records
  checked on 20 Nov 2026 at the URL given in each entry's `note`-free `url` field (proceedings.mlr.press,
  jmlr.org, openreview.net, proceedings.neurips.cc, proceedings.mlsys.org, arxiv.org).
Every fetched record is also logged (title, year) to paper/refs_check.tsv for a final read-through.
"""
import re
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "paper"

DOI = {  # key -> DOI
    "akesson2024random": "10.1016/j.compbiomed.2024.108944",
    "buslaev2020albumentations": "10.3390/info11020125",
    "chen2018reverse": "10.1007/978-3-030-01240-3_15",
    "howard2018universal": "10.18653/v1/P18-1031",
    "hu2020direction": "10.1109/TPAMI.2019.2919616",
    "hu2018direction": "10.1109/CVPR.2018.00778",
    "lin2017feature": "10.1109/CVPR.2017.106",
    "milletari2016vnet": "10.1109/3DV.2016.79",
    "sanin2012shadow": "10.1016/j.patcog.2011.10.001",
    "stauder1999detection": "10.1109/6046.748172",
    "tan2020efficientdet": "10.1109/CVPR42600.2020.01079",
    "vicente2016large": "10.1007/978-3-319-46466-4_49",
    "wang2020moving": "10.1109/TMM.2019.2954752",
    "wang2018stacked": "10.1109/CVPR.2018.00192",
    "wu2018group": "10.1007/978-3-030-01261-8_1",
    "xu2005insignificant": "10.1109/TCSVT.2005.852402",
    "yang2023silt": "10.1109/ICCV51070.2023.01165",
    "zhou2018unetpp": "10.1007/978-3-030-00889-5_1",
    "zhu2021mitigating": "10.1109/ICCV48922.2021.00466",
    "chen2020multitask": "10.1109/CVPR42600.2020.00565",
    "zhu2018bidirectional": "10.1007/978-3-030-01231-1_8",
    "hu2021revisiting": "10.1109/TIP.2021.3049331",
    "cong2023sddnet": "10.1145/3581783.3612482",
    "hu2026unveiling": "10.1007/s11263-026-02744-z",
    "musgrave2020metric": "10.1007/978-3-030-58595-2_41",
    "dodge2019show": "10.18653/v1/D19-1224",
    "kapoor2023leakage": "10.1016/j.patter.2023.100804",
    "chen2023samadapter": "10.1109/ICCVW60793.2023.00361",
    "jie2025shadowadapter": "10.1016/j.eswa.2025.126809",
    "le2018adnet": "10.1007/978-3-030-01216-8_41",
    "zheng2019distraction": "10.1109/CVPR.2019.00531",
    "jie2023rmlanet": "10.1109/TCSVT.2023.3283416",
}

MANUAL = r"""
@misc{wightman2019timm, author={Ross Wightman}, title={PyTorch Image Models}, year={2019}, publisher={GitHub},
  journal={GitHub repository}, doi={10.5281/zenodo.4414861}, howpublished={\url{https://github.com/rwightman/pytorch-image-models}}}
@book{efron1993bootstrap, title={An Introduction to the Bootstrap}, author={Efron, Bradley and Tibshirani, Robert J.},
  publisher={Chapman \& Hall}, address={New York}, year={1993}}
@inproceedings{lee2015deeply, title={Deeply-Supervised Nets}, author={Lee, Chen-Yu and Xie, Saining and Gallagher, Patrick and Zhang, Zhengyou and Tu, Zhuowen},
  booktitle={Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics}, series={Proceedings of Machine Learning Research},
  volume={38}, pages={562--570}, year={2015}, url={https://proceedings.mlr.press/v38/lee15a.html}}
@inproceedings{loshchilov2019decoupled, title={Decoupled Weight Decay Regularization}, author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={International Conference on Learning Representations}, year={2019}, url={https://openreview.net/forum?id=Bkg6RiCqY7}}
@inproceedings{pascanu2013difficulty, title={On the difficulty of training recurrent neural networks}, author={Pascanu, Razvan and Mikolov, Tomas and Bengio, Yoshua},
  booktitle={Proceedings of the 30th International Conference on Machine Learning}, series={Proceedings of Machine Learning Research},
  volume={28}, pages={1310--1318}, year={2013}, url={https://proceedings.mlr.press/v28/pascanu13.html}}
@article{picard2021torch, title={Torch.manual\_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision},
  author={Picard, David}, journal={arXiv preprint arXiv:2109.08203}, year={2021}}
@article{srivastava2014dropout, title={Dropout: A Simple Way to Prevent Neural Networks from Overfitting}, author={Srivastava, Nitish and Hinton, Geoffrey and Krizhevsky, Alex and Sutskever, Ilya and Salakhutdinov, Ruslan},
  journal={Journal of Machine Learning Research}, volume={15}, number={56}, pages={1929--1958}, year={2014}, url={https://jmlr.org/papers/v15/srivastava14a.html}}
@inproceedings{tan2019efficientnet, title={{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks}, author={Tan, Mingxing and Le, Quoc},
  booktitle={Proceedings of the 36th International Conference on Machine Learning}, series={Proceedings of Machine Learning Research},
  volume={97}, pages={6105--6114}, year={2019}, url={https://proceedings.mlr.press/v97/tan19a.html}}
@inproceedings{oliver2018realistic, title={Realistic Evaluation of Deep Semi-Supervised Learning Algorithms}, author={Oliver, Avital and Odena, Augustus and Raffel, Colin and Cubuk, Ekin Dogus and Goodfellow, Ian},
  booktitle={Advances in Neural Information Processing Systems}, volume={31}, year={2018},
  url={https://proceedings.neurips.cc/paper/2018/hash/c1fea270c48e8079d8ddf7d06d26ab52-Abstract.html}}
@inproceedings{bouthillier2021accounting, title={Accounting for Variance in Machine Learning Benchmarks}, author={Bouthillier, Xavier and Delaunay, Pierre and Bronzi, Mirko and Trofimov, Assya and Nichyporuk, Brennan and Szeto, Justin and Sepahvand, Nazanin Mohammadi and Raff, Edward and Madan, Kanika and Voleti, Vikram and Kahou, Samira Ebrahimi and Michalski, Vincent and Arbel, Tal and Pal, Chris and Varoquaux, Ga{\"e}l and Vincent, Pascal},
  booktitle={Proceedings of Machine Learning and Systems}, volume={3}, year={2021},
  url={https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html}}
@article{jie2023adaptershadow, title={{AdapterShadow}: Adapting Segment Anything Model for Shadow Detection}, author={Jie, Leiping and Zhang, Hui},
  journal={arXiv preprint arXiv:2311.08891}, year={2023}}
"""


def fetch(doi):
    req = urllib.request.Request(f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex; charset=utf-8",
                                                                     "User-Agent": "bib-build/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def main():
    out, log, failed = [], ["key\tdoi\tyear\ttitle"], []
    for key, doi in DOI.items():
        try:
            bib = fetch(doi).strip()
        except Exception as e:  # noqa: BLE001
            failed.append(f"{key} {doi}: {e}"); continue
        bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1)}{{{key},", bib, count=1)
        bib = re.sub(r",\s*month\s*=\s*(\{[^}]*\}|\w+)", "", bib)  # month names are not needed and upset tmlr.bst
        t = re.search(r"title\s*=\s*\{(.*?)\},", bib); y = re.search(r"year\s*=\s*\{?(\d{4})", bib)
        log.append(f"{key}\t{doi}\t{y.group(1) if y else '?'}\t{t.group(1) if t else '?'}")
        out.append(bib); time.sleep(0.5)
    (OUT / "refs.bib").write_text("\n\n".join(out) + "\n" + MANUAL)
    (OUT / "refs_check.tsv").write_text("\n".join(log) + "\n")
    print(f"{len(out)} DOI entries fetched, {MANUAL.count('@')} manual entries")
    if failed:
        print("FAILED:\n" + "\n".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
