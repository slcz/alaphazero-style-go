# gonet — a self-play Go AI

An AlphaZero-style go engine that improves by playing against itself.
The engine is built around a small 8-layer transformer neural network
(~6.7M parameters). The engine is trying to be simple to understand
and fast to train and yet deliver reasonable performance. You can try
a trained network checkpoint in the [web](https://slcz.github.io/go/web/) —
it loads the latest 9x9 checkpoint (it023) from HuggingFace by default.

Like Alpha zero, the network has a value head that outputs a board
value and policy head that outputs the probability distribution over
moves. Also as in AlphaZero, the engine improves by playing games
against itself and using Monte Carlo tree search guided by the value
and policy heads.

The engine is designed to be resource-efficient. Starting from a randomly
initialized network, it reaches roughly 1900–2000 Elo on a single
consumer-grade GPU (RTX5090) in about 60 hours (or ~$50 cloud compute cost).

![Selfplay compute vs ladder Elo](charts/elo-vs-flops.png)

*Scaling of the run: accumulated self-play compute per iteration (parsed
from the self-play shards — games × plies/game × sims/ply × FLOPs/forward)
against a chained Elo ladder (50 standard 19×19 games per checkpoint pair,
200 sims/move, it001 anchored at 1000). Per-match records live in
[`checkpoints/matches/`](https://huggingface.co/chriszhang01/gonet/tree/main/checkpoints/matches)
on the model repo.*

## Efficient Selfplay and Training

To keep self-play economical, the engine uses the playout cap
randomization idea (David J. Wu, KataGo). Given a fixed compute budget,
it generates more games by running a full search on only a small number
of plies, searching most others with a smaller budget, and training the
network only on the fully searched plies. The network is trained on the
25% of plies that receive 200 simulations each; the remaining fast plies
use 40 simulations. Average per ply budget is 70 simulations, a puny
number compared with Alphago's 1600 simulations per step.

Instead of AlphaZero's PUCT selection rule, the engine's MCTS search uses
**regularized policy optimization** (Grill et al., 2020): PUCT's
hand-crafted formula approximates the solution to a regularized
policy-optimization objective. Here, $\bar{\pi}$ is computed exactly as

$$
\bar{\pi} = \arg\max_y \left[ q^T y - \lambda_N \,\mathrm{KL}(\pi_\theta, y) \right].
$$

The same $\bar{\pi}$ is the policy-head training target, the distribution
sampled during MCTS descent, and the played move. This has a useful
property: there is no Dirichlet noise, no temperature, no visit-count
target, and no hand-crafted exploration tricks. Because $\bar{\pi}$ is computed
directly from Q-values, it also works well with playout cap randomization
and the fast plies that use a smaller search budget. AlphaZero instead
uses visit counts to estimate Q-values, which intertwines search and learning and
does not perform very well on small MCTS simulation budget.

Also borrowed from Katago idea, the network has a few extra auxiliary training
head. Notably the ownership territory map and score difference.
Terminal state is scored and labeled using Tromp-tylor rules. This implies
that the engine has to clean up (capture) the dead stones before passes.
To train the engine to pass clean and to provide an accurate end of game
ownership and scoring lables for the loss heads, the MCTS utility function
has both the win/loss and score components,

$$
$utility = \lambda \cdot winloss + (1 - \lambda) \cdot scorediff$
$$

During selfplay, $\lambda$ is set to 0.8. For matching, $\lambda$ is set to 1.0 to
maximize the win rate.

The network architecture is a 8 layer, 8 heads, 256 dimensions encoder-only transformer.
2D Rope is used to account for 2D board geometry. In addition, 3x3 convolution layer is
inserted after the upscaling MLP to help with spatial locality. Muon optimizer is used
across the network with learning rate set to 0.01 throughout. The network purposely
supports different board size for mixed board size training and evaluation.

Initial iterations are played on 9x9 board exclusively. Once a basic value/score head
is trained, larger board sizes are gradually introduced. Knowledge trained on smaller
boards transferred well into larger boards, saving computing resource. The final
checkpoint is obtained with merely 15,000 19x19 sized games.

## Build and test

To set up the environment and build the C binaries:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

make -C engine all
make -C engine check

# native NN drivers: self-play, the gate server, and the Stockfish gauntlet
make -C engine build/selfplay build/uci_mux build/sf_gauntlet

python -m pytest tests/ -q
```

To run the self-play → training → evaluation loop:

```sh
mkdir -p data
curl -fL -o /tmp/noob_3moves.epd.zip \
  https://github.com/official-stockfish/books/raw/master/noob_3moves.epd.zip
unzip -p /tmp/noob_3moves.epd.zip > data/book.epd
rm /tmp/noob_3moves.epd.zip

python scripts/selfplay_loop.py \
  --work-dir /data/az-run \
  --book data/book.epd \
  --iterations 20 --games 30000 --device cuda
```

`--init-ckpt` is optional: if it is omitted and `best.pt` does not exist,
the loop creates a random-init checkpoint in the work directory. On
macOS, use `--device mps`. The opening book is used only for evaluation;
self-play generates games by sampling from π̄.

## Training history (rpo-learn)

![rpo-learn lineage: training loss and relative Elo vs self-play effort](docs/rpo_learn_training.png)

Last logged training loss per generation (total, policy, value, and aux
weighted ×0.3) and cumulative gate ΔElo against the previous approved net
(gen1 = 0; band ±√Σσ²), plotted against estimated self-play effort.
Dashed lines mark gen14 (slow-only training), gen18 (π̄ SEARCH), and gen20
(PCR 300/24). Raw per-generation values are in
[the CSV](docs/rpo_learn_training.csv); the source records are
[`rpo-learn/ITERATIONS.md`](https://huggingface.co/chriszhang01/chessnet/blob/main/rpo-learn/ITERATIONS.md).

- **Effort:** new self-play games × mean plies × PCR sims/ply
  (112.5 for 300/50 @ 0.75; gen20 uses 70 for 300/24 @ 0.8333).
- **Excluded:** buffer replay, evaluation games, and companion arms.
  Gen18 counts arm C only (49,150 games); gen20 counts arm A only
  (43,000 games).
- **Mean plies:** derived from recorded rows/positions; gen11/12/15/18
  use a 120-ply fallback.
- **Elo:** cumulative sum of per-generation gate deltas, not an absolute
  rating.

## Web GUI

You can try the trained network in the [web GUI](https://slcz.github.io/chess/).
The net panel defaults to `rpo-learn/gen20.onnx`, the final checkpoint
of the 20-iteration `rpo-learn` lineage.

```sh
# ORT-web runtime, ~42 MB, fetched once into web/vendor/
sh web/get_vendor.sh

# Emscripten (once, macOS example)
git clone https://github.com/emscripten-core/emsdk.git ~/dev/emsdk
cd ~/dev/emsdk && ./emsdk install latest && ./emsdk activate latest
cd -

# Build wasm. On macOS the wrapper handles emsdk + Python 3.12:
web/build.sh
# On Linux, source the emsdk env and call make directly:
# source ~/dev/emsdk/emsdk_env.sh && make -C engine wasm
```

To serve the web page:

```sh
python3 web/serve.py
```
