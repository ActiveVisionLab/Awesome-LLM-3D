
# Awesome-LLM-3D [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

#### Curated by [Xianzheng Ma](https://xianzhengma.github.io/) and [Yash Bhalgat](https://yashbhalgat.github.io/)

---
🔥 Here is a curated list of papers about 3D-Related Tasks empowered by Large Language Models(LLMs). 
It contains frameworks for 3D-LLM training, tools to deploy LLM, courses and tutorials about 3D-LLM and all publicly available LLM checkpoints and APIs.

## Table of Content

- [Awesome-LLM-3D ](#awesome-llm-3D)
  - [3D Understanding](#3d-understanding])
  - [3D Reasoning](#3d-reasoning)
  - [3D Generation](#3d-generation)
  - [3D Embodied Agent](#3d-embodied-agent)
  - [3D Benchmarks](#3d-benchmarks)
  - [Contributing](#contributing)

## 3D Understanding

|  ID |       Keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 1 |     3D-LLM     |      UCLA    | [3D-LLM: Injecting the 3D World into Large Language Models](https://arxiv.org/pdf/1706.03762.pdf)                                                                                                                      |   NeurIPS'2023|  [github](https://github.com/UMass-Foundation-Model/3D-LLM) |
| 3 |         -         |      -      | [From Language to 3D Worlds: Adapting Language Model for Point Cloud Perception](https://openreview.net/forum?id=H49g8rRIiF)                                                              |    -     |
| 4 |       LLM-Grounder       |      U-Mich      | [LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent](https://arxiv.org/pdf/2309.12311.pdf)              |     arxiv     |  [github](https://github.com/sled-group/chat-with-nerf) |
| 5 |     Chat-3D     |      ZJU     | [Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes](https://arxiv.org/pdf/2308.08769v1.pdf)                                                          |  arxiv      |  [github](https://github.com/Chat-3D/Chat-3D)|
| 6 |         3D-VisTA          |      BIGAI      | [3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment](https://arxiv.org/abs/2308.04352)                                                           |    ICCV‘2023  | [github]() |
| 7 |          LEO          |      BIGAI      | [An Embodied Generalist Agent in 3D World](https://arxiv.org/pdf/2311.12871.pdf)                                                           |    -  |  [github](https://github.com/embodied-generalist/embodied-generalist) |
| 9 |       OpenScene       |      ETHz      | [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/pdf/2211.15654.pdf)                                                             |   CVPR’2022  | [github]() |
| 10 | LERF |     UC Berkeley     | [LERF: Language Embedded Radiance Fields](https://arxiv.org/pdf/2303.09553.pdf)                                                   | ICCV‘2023   | [github]() |
| 11 |       ViewRefer       |      CUHK      | [ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding](https://arxiv.org/pdf/2303.16894.pdf)                                                                                               |ICCV'2023 |[github]() |
| 12 |  OpenNerf |    -    | [OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views](https://openreview.net/pdf?id=SgjAojPKb3)                                                                                | | [github]() |
| 13 |         Contrastive Lift         |     Oxford-VGG     | [Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion](https://arxiv.org/pdf/2306.04633.pdf)                                                                                        |   NeurIPS'2023|[github]() |
| 14 |         CLIP2Scene         |      HKU      | [CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP](https://arxiv.org/pdf/2301.04926.pdf)                                                                                        |    CVPR'2023<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F17dd3555fd1ccf1141cf984347fa1b3fd6b009ca%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 15 |         PointLLM         |      CUHK      | [PointLLM: Empowering Large Language Models to UnderstandPoint Clouds](https://arxiv.org/pdf/2308.16911.pdf)                                                                             |    -<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F80d0116d77beeded0c23cf48946d9d10d4faee14%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 16 |        -        |      MIT      | [Leveraging Large (Visual) Language Models for Robot 3D Scene Understanding](https://arxiv.org/pdf/2209.05629.pdf)                                                                      |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F2f3efe44083af91cef562c1a3451eee2f8601d22%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 17 |        PointBind       |      CUHK     | [Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following](https://arxiv.org/pdf/2309.00615.pdf)             |    <br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F002c256d30d6be4b23d365a8de8ae0e67e4c9641%3Ffields%3DcitationCount&query=%24.citationCount&label=citation) |
| 18 |        PLA        |     HKU    | [PLA: Language-Driven Open-Vocabulary 3D Scene Understanding](https://arxiv.org/pdf/2211.16312.pdf)                                                                 |CVPR'2023<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F68f141724814839d556a989646194be88641b143%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 19 |         UniT3D         |      TUM     | [UniT3D: A Unified Transformer for 3D Dense Captioning and Visual Grounding](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_UniT3D_A_Unified_Transformer_for_3D_Dense_Captioning_and_Visual_ICCV_2023_paper.pdf)                                                                          |   ICCV'2023<br>![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F1b6e810ce0afd0dd093f789d2b2742d047e316d5%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 20 |        CG3D        |      JHU      | [CLIP goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition](https://arxiv.org/pdf/2303.11313.pdf)                                                                                                 |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fb3848d32f7294ec708627897833c4097eb4d8778%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 21|        UniHSI      |      Shanghai AI Lab     | [Unified Human-Scene Interaction via Prompted Chain-of-Contacts](https://arxiv.org/pdf/2309.07918.pdf)                                                                                                 |   -<br> ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fab0e3d3e4d42369de5933a3b4c237780b41c0d77%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)     |
| 22 | JM3D-LLM | Xiamen University | [JM3D & JM3D-LLM: Elevating 3D Representation with Joint Multi-modal Cues](https://arxiv.org/pdf/2310.09503v2.pdf)                                        | ![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F7cbc2a7843411a1768ab762930707af0a3c33a19%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 23 |     Open-Fusion     |      -     | [Open-Fusion: Real-time Open-Vocabulary 3D Mapping and Queryable Scene Representation](https://arxiv.org/pdf/2310.03923.pdf)                                                                            |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fd766bffc357127e0dc86dd69561d5aeb520d6f4c%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 24 |         3D-CLR         |      MIT      | [3D Concept Learning and Reasoning from Multi-View Images](https://arxiv.org/pdf/2204.02311.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 25 |         RT-2         |     Google-DeepMind     | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/pdf/2307.15818.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 25 |         MeshGPT         |     TUM     | [MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers](https://arxiv.org/pdf/2311.15475.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 26 |         ShapeGPT         |     Fudan University     | [ShapeGPT: 3D Shape Generation with A Unified Multi-modal Language Model](https://arxiv.org/pdf/2311.17618.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 27 |         Transcribe3D         |     TTI, Chicago     | [Transcribe3D: Grounding LLMs Using Transcribed Information for 3D Referential Reasoning with Self-Corrected Finetuning](https://openreview.net/forum?id=7j3sdUZMTF&noteId=BqwDAp4cY1)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 28 |         -         |     DeepMind & UCL     | [Evaluating VLMs for Score-Based, Multi-Probe Annotation of 3D Objects](https://arxiv.org/pdf/2311.17851.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 29 |         LLMR         |     MIT, RPI & Microsoft     | [LLMR: Real-time Prompting of Interactive Worlds using Large Language Models](https://arxiv.org/pdf/2309.12276.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 30 |         DreamLLM         |     MEGVII & Tsinghua     | [DreamLLM: Synergistic Multimodal Comprehension and Creation](https://arxiv.org/pdf/2309.11499.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 31 |         -         |     MIT    | [Leveraging Large (Visual) Language Models for Robot 3D Scene Understanding](https://arxiv.org/pdf/2209.05629.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 31 |         LLM-Planner         |     The Ohio State University    | [LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models](https://arxiv.org/pdf/2212.04088.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 31 |         LL3DA        |     Fudan University    | [LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning](https://arxiv.org/pdf/2311.18651.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 31 |         SayPlan        |     QUT Centre for Robotics    | [SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://https://arxiv.org/pdf/2307.06135.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|
| 31 |         3D-GPT        |     QANU   | [3D-GPT: PROCEDURAL 3D MODELING WITH LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.12945.pdf)                                                                                                  |![Dynamic JSON Badge](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F094ff971d6a8b8ff870946c9b3ce5aa173617bfb%3Ffields%3DcitationCount&query=%24.citationCount&label=citation)|

## 3D Reasoning
|  ID |       keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 2 |       3D-CLR      |      UCLA     | [3D Concept Learning and Reasoning from Multi-View Images](https://arxiv.org/pdf/2303.11327.pdf)                                                 |   CVPR'2023  | [github](https://github.com/evelinehong/3D-CLR-Official) |


## 3D Generation
|  ID |       keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 

## 3D Embodied Agent

## 3D Benchmarks
|  ID |       keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 8 |     ScanQA     |      RIKEN AIP    | [ScanQA: 3D Question Answering for Spatial Scene Understanding](https://arxiv.org/pdf/2112.10482.pdf)                                                                                                        | CVPR'2023| [github]() |

## Contributing

This is an active repository and your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for LLM, you could vote for them by adding 👍 to them.

---

If you have any question about this opinionated list, do not hesitate to contact me xianzheng@robots.ox.ac.uk.


