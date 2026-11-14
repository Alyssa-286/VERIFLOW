# Provenance: OSF commitments, public history and private originals

## Pre-registration

- OSF registration https://osf.io/bs3wq/, **DOI 10.17605/OSF.IO/BS3WQ**, registered 2026-11-07T06:02:21Z, public and approved (checked 2026-11-14 via the OSF API; D32). Project: https://osf.io/w8scx/.
- The registration's "Other" field commits to **24 files by SHA-256** (20 at the frozen tag `prereg-v1`, 4 at the post-freeze commit of Addendum 1) and to the digest of `configs/grid/` (`39815e7ba24ddb8cf55b3c20d391cf80d0a2e8e756e0646120fa81fc561352a7`).
- **All 24 file hashes and the grid digest verify in this public repository** (procedure below).

## Why commit ids differ from those on OSF

The registration cites commit ids of the original private repository (for example `prereg-v1` = `8ff68a3…`). Before publication, private cluster identifiers (login host name, private IP addresses, account name) were removed from the whole history of three files: `docs/CLUSTER_POLICY.md`, `docs/HPC_RUNBOOK.md` and one version of `scripts/export_run_records.py`. The history was rewritten only by these string replacements (`git filter-branch --tree-filter`); no other file in any commit changed, and none of the 24 committed files contains a replaced string. Rewriting changes every commit id, so the map below links each original commit to its public counterpart. The original repository with its unmodified history is kept private by the author and can be made available for verification on request (D34).

Run manifests in `results/grid/run_records/` record the original commit of the code that ran (`540206cae0daf19dc86a5625b3b30635f9d3f7c5`), which corresponds to public commit `293d468cb5fb635ca8ce98fca717689fe5cf7595`.

## Verify the OSF commitments

```bash
git show <public commit for 8ff68a3>:docs/PREREGISTRATION.md | sha256sum     # c060d163…e486 on OSF
git show <public commit for 500dfb1>:experiments/analyse_grid.py | sha256sum  # value listed on OSF
```

The tag `prereg-v1` points to the public counterpart of `8ff68a3`.

## Release archive

`veriflow_grid_verifier_logs.tar.gz` (every verifier decision of the 96 grid runs): sha256 `c1fbb1f97631c645cfadd5b165ee48122ee187c0b5a72e1e4ca9df0fd31c7236`. Per-file hashes of the originals are in `results/grid/run_records/INDEX.json`.

## Commit map (original private → public)

| Original | Public |
|---|---|
| `7a810d270ce072965206fb81fa3799cf4970ae4d` | `7a810d270ce072965206fb81fa3799cf4970ae4d` |
| `e60a2f16bdac21ad0753b999c4c268db6df162f2` | `29b80847d5a18cfa3368af654b342333695e2d36` |
| `a503e7a4391b8e4e806db6f47755c7f73c01c649` | `5f5d004f880cc0e296cbeca04292bbafe338a7c2` |
| `9b41884b6784789e369aed9f402d70db381b071e` | `27cef46d239ffc6809e8914600b2605ae247e3de` |
| `4e812da186ba67bdc0a9117878a736d69fb63e0e` | `3a0ec6797bcd32bb46171beb059bd895df80fb43` |
| `ab02b2169d919ae63f81a800c79ccd14523f3281` | `57bbec86c801dc820b396c7f623f1092d65a7b41` |
| `6c04e29d040fc06525c49ad522b95a8b4c046928` | `9393956051e4d2ee4fad807f149801744d3707d8` |
| `94b7ac58f528ecf4b1423c1f98667d47d332bd27` | `e82a424f7c5d6e214a6aebeb9f84c85e1680cb1f` |
| `47fa0d2749478e1ade3a65c8e6864061816930bd` | `35dedaab3ede73afe5395fbbbf122317e2f45dec` |
| `7ffd8c535372b4930d7485b78727e87a196fcb5d` | `1b55292fb241c13a1073789417529de4ad336f8b` |
| `36de3956ebaad7eda2e433b285742b302c5187f5` | `2cf853d98f241bcbce6dccd71560dae03b02b9f3` |
| `442e06e97056f77e2ab93822c99e8896d0426d90` | `3356c35bbf1b7b02d99d282a4dd3f05350b6152c` |
| `3517a3330c41dffd24b6fbd55190fe4e8633312e` | `8cdc2ff20d16e18b3c72449208e61ffaa45e6b4e` |
| `16483368f9b391cf8ddcaf83d7caf5bb519e5c00` | `99d8ba23c0ca04870b205fe12a9ad7da11c2fcde` |
| `b4afaead3e4fd3f28823b190af5f8f3da69868ad` | `2f33fb14b3cd4bcb9553e5bb5c981d0e13e454e8` |
| `7eaf051fd3cc7f77b1289a33104f13c5a7600e1f` | `cebbd92cdbbc55d9daf6aaed2642d21e18a86cc9` |
| `ffa27a74cdf190dd0d3598b8a628d3fa282046d1` | `4d7628906d7e09292b4edb379a8ffc5d2f93f31f` |
| `ff6ea22f4e78f832927a58d374134a9388f0edce` | `0f1f8c4b1118a0f7fe8c56641aa98d7afe63eb48` |
| `508fefa9e6dfcc754ab6b5023430716a1ec1690d` | `a45568ddcc0c1d879b712875e332aef59ac49e28` |
| `bcf0e24cb818405005ad83270c20fe21074e94fa` | `68495fe111b0fccd150ec80a81766d8deea0c3d4` |
| `d477f0b65dbb0209d1ff486fc83877909ca805cd` | `445b16307a9faabd06b858800c8350bd22f3a26d` |
| `24c49a389abeeb273b9c759d196c51280cd4bd29` | `cfb157e1818d887b9f9f1ff4ac15e6a2c174a39e` |
| `9849f5d31a5b0b21097eff564a4aa43ba17fb75e` | `df125c021c2c9bf5c1850546a33e284580e60b2c` |
| `b0a0b089993a0a1ebfaceb18d08ef8d29e960414` | `03cc70a072fb58579eda95511595faf4b4bce5d7` |
| `17e94313f42b7a7e7782b51332fe111ea4c6e92c` | `972fef74ffe460a759304df959798c1c299bad5e` |
| `7b339012afea9e2e3431bb45e9ac4c84c5930ede` | `55d1d0391205f758e369dede144c90731284e378` |
| `b4fe69bea016ad3a7e18ec76b7df328658192bef` | `cdbc9729789dc8cfb7e4ced79898ea37d102ed37` |
| `c9349304072ab71bf2530e9a9b25a07cdfb9d734` | `96577bbe44c5e65323964444360b4c04104936d6` |
| `e9b8b608a2719ddda063ee473062bda9c00438af` | `c702bc820b83e028ab4d55dcc92f125d2cccdd58` |
| `48649efae3493b9bcbc901ab763d9475e3664ae7` | `e51bd73560f953284d6489d3afa272cb83517f77` |
| `a6175032181c3229371d48fd508d3ec61516b879` | `7552ab5e098403f79cf4c0c4a285f0cda9601b3d` |
| `0633bd0bf3a973a542184b45db326784e7d0c4b1` | `109a5fec8241357d5b077d9deef7496b5a199927` |
| `9c1609806e7ee264ba8a0ef2f26dae80ec5c25e5` | `464786ecc1def12df1b4750bb07e50b0a760797c` |
| `6ba5fb12dff2ef6df3aa13131f2ca1f18831cb7c` | `e935feebba658f42ae3422aebf43f3fff3234a19` |
| `663e41c546ad73041707c843672a2d1b6d630f5c` | `f468859414a166c209a0f9301aa731bc8650625c` |
| `c97859979aadf10021e96647569d69fbe5a45bec` | `c9b06e5ea68ca3e0443525b189ea3add06b3b8c0` |
| `eb447e083d1aeec0696628087ff1d7420c82d2d5` | `2ef93b4a06d714351dc0be479a6806a9a9d39117` |
| `762aa72606635fb3595dcf6f14443076f7c37f45` | `dda01931c37b9a8cf14dea0c3bba7446041b1901` |
| `8ff68a3a0c620d9bf2daf4500c6626db1c0754d6` | `e6cac4a915c57909ad0de18d757a33bfab9324b8` |
| `b8615487d977d27c6bd9defe23d086f00e231a8b` | `10d28c9c508ca494f5e07ec2e865fad87c2f06fa` |
| `053b20f1373900af15f9efb9f43e818fc3745f9d` | `395f7dacca506b4fc01328b075a7a951425ad902` |
| `5271f91573870975fc10064a320384ea8356fdc3` | `4214f05409b71f7fbbf82b419fa99be315bb4495` |
| `75a819de899c9117cf0fa32a224273cec6005b58` | `7cf93e4ce443ece5c9de762d9efc4255b25cfb43` |
| `a03c4cc9242db592b965f58c8c7c5e90afd777e7` | `ed2a0665d6741402edef79663941801956bdd54f` |
| `500dfb12be5a4b9b9ed401b173063066fa12db26` | `ef059d5dd153d05f97f23b7a22afbde6800623c2` |
| `540206cae0daf19dc86a5625b3b30635f9d3f7c5` | `293d468cb5fb635ca8ce98fca717689fe5cf7595` |
| `655cff060de4f8a37e10ae6da3a449f35b68715c` | `e644d720b713a196bdde4ddb6f3ed356512bb82d` |
| `6ac1e81a910363b254e38aa358e3c7e7f6c04424` | `3107fa672eb157162b86be644b2b1b377671dca0` |
| `0d850fa997c378022fc58386e8e254df162bd6fb` | `e1009358c87e05038644b71266aeb28af0964563` |
| `4405dffa17e7d0e40cf5e9cc995397bd8c4cbfcd` | `e4a513480a859b7049e50350d37cb0364b8bbb1e` |
