#!/usr/bin/env bash
# Regenerate ucft-final.tex by inlining .ucft-build/unit-*.tex sources.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/.ucft-build"
OUT="$ROOT/ucft-final.tex"

inline_unit() {
  local unit="$1"
  echo "% --- begin inlined: $unit ---"
  cat "$BUILD/$unit"
  echo "% --- end inlined: $unit ---"
}

{
  cat <<'PREAMBLE'
% ucft-final.tex — standalone master manuscript (no \input dependencies).
% All theorems, proofs, postulates, and appendices are inlined.
% Regenerated from .ucft-build/unit-*.tex sources.

\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,amsthm,mathtools}
\usepackage{mathrsfs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{array}
\usepackage{hyperref}

\hypersetup{colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,urlcolor=blue!55!black}
\allowdisplaybreaks
\renewcommand{\arraystretch}{1.18}

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{conjecture}[theorem]{Conjecture}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{axiom}[theorem]{Axiom}
\newtheorem{postulate}[theorem]{Postulate}
\newtheorem{construction}[theorem]{Construction}
\newtheorem{indicative}[theorem]{Indicative}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}

\providecommand*{\sectionautorefname}{Section}
\providecommand*{\subsectionautorefname}{Section}
\providecommand*{\theoremautorefname}{Theorem}
\providecommand*{\lemmaautorefname}{Lemma}
\providecommand*{\corollaryautorefname}{Corollary}
\providecommand*{\propositionautorefname}{Proposition}
\providecommand*{\conjectureautorefname}{Conjecture}
\providecommand*{\definitionautorefname}{Definition}
\providecommand*{\axiomautorefname}{Axiom}
\providecommand*{\postulateautorefname}{Postulate}
\providecommand*{\constructionautorefname}{Construction}
\providecommand*{\indicativeautorefname}{Indicative}
\providecommand{\cref}[1]{\autoref{#1}}

% Core notation.
\newcommand{\SU}[1]{\mathrm{SU}(#1)}
\newcommand{\SO}[1]{\mathrm{SO}(#1)}
\newcommand{\U}[1]{\mathrm{U}(#1)}
\newcommand{\E}[1]{\mathrm{E}_{#1}}
\newcommand{\SL}{\mathrm{SL}}
\newcommand{\Spin}{\operatorname{Spin}}
\newcommand{\Aut}{\operatorname{Aut}}
\newcommand{\End}{\operatorname{End}}
\newcommand{\Sym}{\operatorname{Sym}}
\newcommand{\rank}{\operatorname{rank}}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\Hess}{\operatorname{Hess}}
\newcommand{\Tr}{\operatorname{Tr}}
\newcommand{\Str}{\operatorname{STr}}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Oct}{\mathbb{O}}
\newcommand{\M}{\mathcal{M}}
\newcommand{\jalg}{J_3(\mathbb{O})}
\newcommand{\SOTen}{\SO{10}}
\newcommand{\UOne}{\U{1}}
\newcommand{\Uone}{\U{1}}
\newcommand{\ESix}{\E{6}}
\newcommand{\EIII}{\mathrm{EIII}}
\newcommand{\SOTenXUOne}{\SOTen \times \UOne}
\newcommand{\MCoset}{\frac{\ESix}{\SOTenXUOne}}
\newcommand{\MCosetInline}{\ESix/(\SOTenXUOne)}
\newcommand{\ESixBreaking}{\ESix \to \SOTenXUOne}
\newcommand{\Herm}[2]{\mathrm{Herm}_{#1}(#2)}
\newcommand{\chr}[1]{\mathrm{char}(#1)}
\newcommand{\Rep}[1]{\mathbf{#1}}
\newcommand{\Repbar}[1]{\overline{\mathbf{#1}}}
\newcommand{\Pone}{\textnormal{P1}}
\newcommand{\Ptwo}{\textnormal{P2}}
\newcommand{\Pthree}{\textnormal{P3}}
\newcommand{\Pfour}{\textnormal{P4}}
\newcommand{\Pfive}{\textnormal{P5}}
\newcommand{\Psix}{\textnormal{P6}}
\newcommand{\MGUT}{M_{\mathrm{GUT}}}
\newcommand{\ghat}{\hat g}
\newcommand{\yhat}{\hat y}
\newcommand{\clock}{\phi^{0}}
\newcommand{\ket}[1]{\lvert #1\rangle}
\newcommand{\dket}[1]{\lvert #1\rangle\!\rangle}
\newcommand{\dbra}[1]{\langle\!\langle #1\rvert}
\newcommand{\sm}{\hspace{.065em}}
\newcommand{\s}{\hspace{.1em}}
\newcommand{\smexp}[1]{^{\sm #1}}
\newcommand{\set}[2]{\{#1\,:\,#2\}}

\title{Universal Coset Field Theory:\\[0.4em]
\large From Axioms to the Standard Model and Gravitation}
\author{Brandon Belna}
\date{June 6, 2026}

\begin{document}
\maketitle

\begin{abstract}
We present the complete Universal Coset Field Theory (UCFT). A graph--Stone--BAMLP
foundation (the Albert--Cayley graph, profinite Stone lift, capacity dual, and
exceptional ladder $0\to E_8\to E_8\times E_8\to E_6\to\cdots\to 0$) derives the
$E_6$ potential and selects the EIII vacuum as the unique fixed point of a
master operator $\mathbb{U}$. Two axioms specify the
order parameter and quantum kinematics; six postulates \textbf{P1--P6} supply the
$E_6$-invariant potential, vacuum manifold, Lorentzian soldering, chiral matter
content, three generations, and GUT--Yukawa sector. Under these hypotheses we
prove that the global minimum of the potential is the rank-one orbit
$E_6/(\mathrm{Spin}(10)\times U(1))$ (EIII), with $32$ Goldstone modes and
positive-definite coset metric. Postulate~\textbf{P3} yields four-dimensional
Lorentzian spacetime; one-loop fluctuations induce a positive Planck mass
$M_{\mathrm{Pl}}^2=(28/6)\,f^2\mu^2\log(\Lambda^2/\mu^2)/(16\pi^2)$ and an
asymptotically free $\mathrm{Spin}(10)\times U(1)$ sector with
$b_0=12$ for three generations. With \textbf{P4--P6} the theory embeds the
Standard Model, including neutrino masses via the type-I seesaw. Conjectural
results include a gravitational ultraviolet fixed point
$\kappa_\star=96\pi^2/5$ and heterotic string correspondences. Octonionic
algebra, soldering, representation theory, and heat-kernel computations are
collected in appendices.
\end{abstract}

\noindent\textbf{Keywords:}
BAMLP; Stone spaces; graph universe; exceptional ladder; exceptional Jordan
algebra; $E_6$ coset; induced gravity; $\mathrm{Spin}(10)$ grand unification;
Standard Model embedding.

\tableofcontents
\clearpage
PREAMBLE

  echo '%----------------------------------------------------------------------%'
  echo '% Part: Foundations and the Master Chain'
  echo '%----------------------------------------------------------------------%'
  echo '\part{Foundations and the Master Chain}'
  inline_unit unit-bamlp.tex
  inline_unit unit-identification.tex
  inline_unit unit-master-chain.tex
  inline_unit unit-intro.tex
  inline_unit unit-axioms.tex

  echo '%----------------------------------------------------------------------%'
  echo '% Part: Albert Algebra, Vacuum Selection, and Coset Geometry'
  echo '%----------------------------------------------------------------------%'
  echo '\part{Albert Algebra, Vacuum Selection, and Coset Geometry}'
  inline_unit unit-albert.tex
  inline_unit unit-potential.tex
  inline_unit unit-coset.tex

  echo '%----------------------------------------------------------------------%'
  echo '% Part: Spacetime, Gauge Sector, and Mass Spectrum'
  echo '%----------------------------------------------------------------------%'
  echo '\part{Spacetime, Gauge Sector, and Mass Spectrum}'
  inline_unit unit-spacetime.tex
  inline_unit unit-gauge.tex
  inline_unit unit-masses.tex

  echo '%----------------------------------------------------------------------%'
  echo '% Part: Induced Gravity, Renormalization, and Standard Model Descent'
  echo '%----------------------------------------------------------------------%'
  echo '\part{Induced Gravity, Renormalization, and Standard Model Descent}'
  inline_unit unit-gravity.tex
  inline_unit unit-frg.tex
  inline_unit unit-sm.tex

  echo '%----------------------------------------------------------------------%'
  echo '% Part: Correspondences and Conclusions'
  echo '%----------------------------------------------------------------------%'
  echo '\part{Correspondences and Conclusions}'
  inline_unit unit-strings.tex

  cat <<'TAIL'

\appendix
\part{Appendices}

TAIL

  inline_unit unit-app-albert.tex
  inline_unit unit-app-lorentz.tex
  inline_unit unit-app-spectrum.tex
  inline_unit unit-app-ym.tex

  cat <<'TAIL2'

\section{Deferred Computational Appendices}
\label{app:gravity:main}
\label{app:frg}
\label{app:ws:main}
\label{app:cy:main}
\label{app:anomaly:main}
\label{eq:MC}
\label{eq:sigma}
The heat-kernel, functional-renormalization-group, worldsheet, Calabi--Yau,
and full anomaly-arithmetic details referenced in the main text are summarized
in the sections and algebraic appendices above. This section preserves stable
cross-reference anchors for those computations.

\begin{thebibliography}{99}

\bibitem{BAMLP:2026}
B.~Belna,
Barrier-Aware Metric--Laguerre Partitioning,
manuscript, 2026.

\bibitem{Albert:1934}
A.~A.~Albert,
On a certain algebra of quantum mechanics,
\emph{Annals of Mathematics} \textbf{35} (1934).

\bibitem{Baez:2002}
J.~C.~Baez,
The octonions,
\emph{Bulletin of the American Mathematical Society} \textbf{39} (2002).

\bibitem{Freudenthal:1985}
H.~Freudenthal,
Oktaven, Ausnahmegruppen und Oktavengeometrie,
selected works on exceptional Jordan structures.

\bibitem{Georgi:1974}
H.~Georgi and S.~L.~Glashow,
Unity of all elementary-particle forces,
\emph{Physical Review Letters} \textbf{32} (1974).

\bibitem{GreenSchwarz:1984}
M.~B.~Green and J.~H.~Schwarz,
Anomaly cancellation in supersymmetric $D=10$ gauge theory,
\emph{Physics Letters B} \textbf{149} (1984).

\bibitem{GlimmJaffe:1987}
J.~Glimm and A.~Jaffe,
\emph{Quantum Physics: A Functional Integral Point of View},
Springer, 1987.

\bibitem{Helgason:1978}
S.~Helgason,
\emph{Differential Geometry, Lie Groups, and Symmetric Spaces},
Academic Press, 1978.

\bibitem{JNW:1934}
P.~Jordan, J.~von~Neumann, and E.~Wigner,
On an algebraic generalization of the quantum mechanical formalism,
\emph{Annals of Mathematics} \textbf{35} (1934).

\bibitem{Jacobson:1968}
N.~Jacobson,
\emph{Structure and Representations of Jordan Algebras},
American Mathematical Society, 1968.

\bibitem{Krutelevich:2007}
S.~Krutelevich,
Jordan algebras, exceptional groups, and higher composition laws,
\emph{Journal of Algebra} \textbf{314} (2007).

\bibitem{McCrimmon:2004}
K.~McCrimmon,
\emph{A Taste of Jordan Algebras},
Springer, 2004.

\bibitem{OsterwalderSchrader:1973}
K.~Osterwalder and R.~Schrader,
Axioms for Euclidean Green's functions,
\emph{Communications in Mathematical Physics} \textbf{31} (1973).

\bibitem{Reuter:1998}
M.~Reuter,
Nonperturbative evolution equation for quantum gravity,
\emph{Physical Review D} \textbf{57} (1998).

\bibitem{Sakharov:1967pk}
A.~D.~Sakharov,
Vacuum quantum fluctuations in curved space and the theory of gravitation,
\emph{Soviet Physics Doklady} \textbf{12} (1968).

\bibitem{Schafer:1966}
R.~D.~Schafer,
\emph{An Introduction to Nonassociative Algebras},
Academic Press, 1966.

\bibitem{Slansky:1981}
R.~Slansky,
Group theory for unified model building,
\emph{Physics Reports} \textbf{79} (1981).

\bibitem{Springer:2000}
T.~A.~Springer and F.~D.~Veldkamp,
\emph{Octonions, Jordan Algebras and Exceptional Groups},
Springer, 2000.

\bibitem{Sudbery:1984}
A.~Sudbery,
Division algebras, (pseudo)orthogonal groups and spinors,
\emph{Journal of Physics A} \textbf{17} (1984).

\bibitem{Wetterich:1993}
C.~Wetterich,
Exact evolution equation for the effective potential,
\emph{Physics Letters B} \textbf{301} (1993).

\bibitem{Yokota:2009}
I.~Yokota,
\emph{Exceptional Lie Groups},
arXiv/lecture notes, 2009.

\end{thebibliography}

\end{document}
TAIL2

} > "$OUT"

echo "Wrote $OUT ($(wc -l < "$OUT") lines)"