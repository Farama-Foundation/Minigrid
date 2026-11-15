---
firstpage:
lastpage:
---

## BabyAI Environments

The environments listed below are implemented in the [minigrid/envs/babyai](/minigrid/envs/babyai)
directory. They were imported from the [BabyAI](https://github.com/mila-iqia/babyai) project and
are the levels described in its ICLR 2019 paper. Each level is a Minigrid grid-world paired with an
instruction drawn from the *Baby Language* grammar - "go to the red ball", "pick up the grey box
behind you, then open a door" - which the agent reads from the `mission` field of the observation,
and with a verifier that decides when that instruction has been carried out. The levels isolate
different competencies, from a single room with grey distractors and no language (`GoToRedBallGrey`)
up to `BossLevel`, a superset of all the others whose instruction can be any sentence of the
grammar. The purpose of the collection is research on grounded language learning.

```{raw} html
   :file: list.html
```

```{toctree}
:hidden:
:caption: BabyAI Environments

GoToRedBallGrey
GoToRedBall
GoToRedBallNoDists
GoToObj
GoToLocal
GoTo
GoToImpUnlock
GoToSeq
GoToRedBlueBall
GoToDoor
GoToObjDoor
Open
OpenRedDoor
OpenDoor
OpenTwoDoors
OpenDoorsOrder
Pickup
UnblockPickup
PickupLoc
PickupDist
PickupAbove
PutNextLocal
PutNext
Unlock
UnlockLocal
KeyInBox
UnlockPickup
BlockedUnlockPickup
UnlockToUnlock
ActionObjDoor
FindObjS5
KeyCorridor
OneRoomS8
MoveTwoAcross
Synth
SynthLoc
SynthSeq
MiniBossLevel
BossLevel
BossLevelNoUnlock

```
