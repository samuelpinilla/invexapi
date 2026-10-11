# References

Works cited in the [README](README.md) and [GUIDE](GUIDE.md). Short labels such as
**[ICLR'24]** in those files link to the entries below.

## Papers this library implements

- <a id="ref-neurips22"></a>**[NeurIPS'22]** S. Pinilla, T. Mu, N. Bourne and
  J. Thiyagalingam. *Improved imaging by invex regularizers with global optima
  guarantees*. Advances in Neural Information Processing Systems 35 (NeurIPS 2022),
  pp. 10780–10794.
  [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/45f7927942098d14e473fc5d000031e2-Abstract.html)
  · [arXiv:2211.10112](https://arxiv.org/abs/2211.10112)
  — the invex regularizers (Lemma 1), the affine invex construction (Lemma 2).
- <a id="ref-iclr24"></a>**[ICLR'24]** S. Pinilla and J. Thiyagalingam. *Global
  optimality for non-linear constrained restoration problems via invexity*.
  International Conference on Learning Representations (ICLR 2024),
  pp. 11990–12027.
  [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3333f5a87d626f8e46af7c4b27363066-Abstract-Conference.html)
  — admissible functions (Def. 4, Thm 3), robust losses (Thm 5), invex TV (Thm 4, App. B.1).
- <a id="ref-trainable25"></a>**[Trainable'25]** S. Pinilla, A. M. Sanabria, J. Bi
  and K. Egiazarian. *What makes neural networks trainable? Invexity as a structural
  design principle in AI*. Research Square preprint rs-7215670/v1, 2025
  — invex layers and their composition (`invexapi.nn`).

## Related work by the authors

- <a id="ref-icassp24"></a>**[ICASSP'24]** S. Pinilla, S.-L. Yeung and
  J. Thiyagalingam. *Global convergence of alternating direction method of
  multipliers for invex objective losses*. IEEE International Conference on
  Acoustics, Speech and Signal Processing (ICASSP 2024), pp. 9361–9365.
  [doi:10.1109/ICASSP48485.2024.10445993](https://doi.org/10.1109/ICASSP48485.2024.10445993)

## Other works the library relies on

- <a id="ref-boyd04"></a>**[Boyd'04]** S. Boyd and L. Vandenberghe. *Convex
  Optimization*. Cambridge University Press, 2004.
  [Book website](https://web.stanford.edu/~boyd/cvxbook/)
  — convexity rules (§3.2) used by `invexapi.rules`.
- <a id="ref-hanson81"></a>**[Hanson'81]** M. A. Hanson. *On sufficiency of the
  Kuhn-Tucker conditions*. Journal of Mathematical Analysis and Applications 80
  (1981), pp. 545–550.
  [doi:10.1016/0022-247X(81)90123-2](https://doi.org/10.1016/0022-247X(81)90123-2)
  — the definition of invexity (convex ⇒ invex).
- <a id="ref-ozaktas96"></a>**[Ozaktas'96]** H. M. Ozaktas, O. Arikan, M. A. Kutay
  and G. Bozdagi. *Digital computation of the fractional Fourier transform*. IEEE
  Transactions on Signal Processing 44(9) (1996), pp. 2141–2150.
  [doi:10.1109/78.536672](https://doi.org/10.1109/78.536672)
  — the fast FrFT in `invexapi.operators`.
- <a id="ref-fasta"></a>**[FASTA]** T. Goldstein, C. Studer and R. Baraniuk. *A field
  guide to forward-backward splitting with a FASTA implementation*. 2014.
  [arXiv:1411.3406](https://arxiv.org/abs/1411.3406)
  — the `FASTA` solver.

<details>
<summary>BibTeX for the papers this library implements</summary>

```bibtex
@inproceedings{pinilla2022invex,
  author    = {Pinilla, Samuel and Mu, Tingting and Bourne, Neil and Thiyagalingam, Jeyan},
  title     = {Improved Imaging by Invex Regularizers with Global Optima Guarantees},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {10780--10794},
  year      = {2022}
}

@inproceedings{pinilla2024global,
  author    = {Pinilla, Samuel and Thiyagalingam, Jeyan},
  title     = {Global Optimality for Non-linear Constrained Restoration Problems via Invexity},
  booktitle = {International Conference on Learning Representations},
  pages     = {11990--12027},
  year      = {2024}
}

@misc{pinilla2025trainable,
  author       = {Pinilla, Samuel and Sanabria, Ana M. and Bi, Jia and Egiazarian, Karen},
  title        = {What Makes Neural Networks Trainable? {I}nvexity as a Structural Design Principle in {AI}},
  howpublished = {Research Square preprint rs-7215670/v1},
  year         = {2025}
}
```

</details>
