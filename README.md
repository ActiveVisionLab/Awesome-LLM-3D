
# Awesome-LLM-3D [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

#### Curated by [Xianzheng Ma](https://xianzhengma.github.io/) and [Yash Bhalgat](https://yashbhalgat.github.io/)

---
🔥 Here is a curated list of papers about 3D-Related Tasks empowered by Large Language Models(LLMs). 
It contains various tasks including 3D understanding, reasoning, generation, and embodied agents.

## Table of Content

- [Awesome-LLM-3D](#awesome-llm-3D)
  - [3D Understanding](#3d-understanding])
  - [3D Reasoning](#3d-reasoning)
  - [3D Generation](#3d-generation)
  - [3D Embodied Agent](#3d-embodied-agent)
  - [3D Benchmarks](#3d-benchmarks)
  - [Contributing](#contributing)



## 3D Understanding

|  ID |       Keywords       |    Institute   | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 1 |     3D-LLM     |      UCLA    | [3D-LLM: Injecting the 3D World into Large Language Models](https://arxiv.org/pdf/2307.12981.pdf)                                                                                                                      |   NeurIPS'2023|  [github](https://github.com/UMass-Foundation-Model/3D-LLM) |
| 2 |         LL3DA        |     Fudan University    | [LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning](https://arxiv.org/pdf/2311.18651.pdf)                                                                                                  |Arxiv|  [github](https://github.com/Open3DA/LL3DA) |
| 3 |       LLM-Grounder       |      U-Mich      | [LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent](https://arxiv.org/pdf/2309.12311.pdf)              |     Arxiv     |  [github](https://github.com/sled-group/chat-with-nerf) |
| 4 |        Point-Bind       |      CUHK     | [Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following](https://arxiv.org/pdf/2309.00615.pdf)             |  Arxiv   |  [github](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM) |
| 5 |         3D-VisTA          |      BIGAI      | [3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment](https://Arxiv.org/abs/2308.04352)                                                           |    ICCV‘2023  | [github]() |
| 6 |          LEO          |      BIGAI      | [An Embodied Generalist Agent in 3D World](https://arxiv.org/pdf/2311.12871.pdf)                                                           |    Arxiv  |  [github](https://github.com/embodied-generalist/embodied-generalist) |
| 7 |       OpenScene       |      ETHz      | [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/pdf/2211.15654.pdf)                                                             |   CVPR’2022  | [github](https://github.com/pengsongyou/openscene) |
| 8 | LERF |     UC Berkeley     | [LERF: Language Embedded Radiance Fields](https://arxiv.org/pdf/2303.09553.pdf)                                                   | ICCV‘2023   | [github](https://github.com/kerrj/lerf) |
| 9 |       ViewRefer       |      CUHK      | [ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding](https://arxiv.org/pdf/2303.16894.pdf)                                                                                               |ICCV'2023 |[github](https://github.com/Ivan-Tang-3D/ViewRefer3D) |
| 10 |         Contrastive Lift         |     Oxford-VGG     | [Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion](https://arxiv.org/pdf/2306.04633.pdf)                                                                                        |   NeurIPS'2023| [github](https://github.com/yashbhalgat/Contrastive-Lift) |
| 11 |         CLIP2Scene         |      HKU      | [CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP](https://arxiv.org/pdf/2301.04926.pdf)                                                                                        |    CVPR'2023 | [github](https://github.com/runnanchen/CLIP2Scene) |
| 12 |         PointLLM         |      CUHK      | [PointLLM: Empowering Large Language Models to UnderstandPoint Clouds](https://arxiv.org/pdf/2308.16911.pdf)                                                                             |   Arxiv |  [github](https://github.com/OpenRobotLab/PointLLM) |
| 13 |        -        |      MIT      | [Leveraging Large (Visual) Language Models for Robot 3D Scene Understanding](https://arxiv.org/pdf/2209.05629.pdf)                                                                      |Arxiv|  [github](https://github.com/MIT-SPARK/llm_scene_understanding) |
| 14 |     Chat-3D     |      ZJU     | [Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes](https://arxiv.org/pdf/2308.08769v1.pdf)                                                          |  Arxiv      |  [github](https://github.com/Chat-3D/Chat-3D)|
| 15 |        PLA        |     HKU    | [PLA: Language-Driven Open-Vocabulary 3D Scene Understanding](https://arxiv.org/pdf/2211.16312.pdf)                                                                 |CVPR'2023|  [github](https://github.com/CVMI-Lab/PLA) |
| 16 |         UniT3D         |      TUM     | [UniT3D: A Unified Transformer for 3D Dense Captioning and Visual Grounding](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_UniT3D_A_Unified_Transformer_for_3D_Dense_Captioning_and_Visual_ICCV_2023_paper.pdf)                                                                          |   ICCV'2023| [github]() |
| 17 |        CG3D        |      JHU      | [CLIP goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition](https://arxiv.org/pdf/2303.11313.pdf)                                                                                                 |Arxiv|  [github](https://github.com/deeptibhegde/CLIP-goes-3D) |
| 18 | JM3D-LLM | Xiamen University | [JM3D & JM3D-LLM: Elevating 3D Representation with Joint Multi-modal Cues](https://arxiv.org/pdf/2310.09503v2.pdf)                                        | ACM MM'2023 |  [github](https://github.com/mr-neko/jm3d) |
| 19 |     Open-Fusion     |      -     | [Open-Fusion: Real-time Open-Vocabulary 3D Mapping and Queryable Scene Representation](https://arxiv.org/pdf/2310.03923.pdf)                                                                            |Arxiv|  [github](https://github.com/UARK-AICV/OpenFusion) |
| 20 |         -         |      -      | [From Language to 3D Worlds: Adapting Language Model for Point Cloud Perception](https://openreview.net/forum?id=H49g8rRIiF)                                                              |    OpenReview     | - |
| 21 |  OpenNerf |    -    | [OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views](https://openreview.net/pdf?id=SgjAojPKb3)                                                                                | OpenReview | [github]() |
| 22 |  - |    KAUST & LIX  | [Zero-Shot 3D Shape Correspondence](https://arxiv.org/abs/2306.03253)                                                                                | Siggraph Asia 2023 | - |

## 3D Reasoning
|  ID |       keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 1 |       3D-CLR      |      UCLA     | [3D Concept Learning and Reasoning from Multi-View Images](https://arxiv.org/pdf/2303.11327.pdf)                                                 |   CVPR'2023  | [github](https://github.com/evelinehong/3D-CLR-Official) |
| 2 |         Transcribe3D         |     TTI, Chicago     | [Transcribe3D: Grounding LLMs Using Transcribed Information for 3D Referential Reasoning with Self-Corrected Finetuning](https://openreview.net/pdf?id=7j3sdUZMTF)                                                                                                  |CoRL'2023|  [github]() |


## 3D Generation
|  ID |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 1 |         3D-GPT        |     ANU   | [3D-GPT: PROCEDURAL 3D MODELING WITH LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2310.12945.pdf)                                                                                                | Arxiv  |  [github]() |
| 2|         MeshGPT         |     TUM     | [MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers](https://arxiv.org/pdf/2311.15475.pdf)                                                                                                  |Arxiv |  [project](https://nihalsid.github.io/mesh-gpt/) |
| 3 |         ShapeGPT         |     Fudan University     | [ShapeGPT: 3D Shape Generation with A Unified Multi-modal Language Model](https://arxiv.org/pdf/2311.17618.pdf)                                                                                                  |Arxiv|  [github](https://github.com/OpenShapeLab/ShapeGPT) |
| 4 |         DreamLLM         |     MEGVII & Tsinghua     | [DreamLLM: Synergistic Multimodal Comprehension and Creation](https://arxiv.org/pdf/2309.11499.pdf)                                                                                                  |Arxiv|  [github](https://dreamllm.github.io/) |
| 5 |         LLMR         |     MIT, RPI & Microsoft     | [LLMR: Real-time Prompting of Interactive Worlds using Large Language Models](https://arxiv.org/pdf/2309.12276.pdf)                                                                                                  |Arxiv|  [github]() |
| 6 |      ChatAvatar      |       Deemos Tech            | [DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance](https://dl.acm.org/doi/abs/10.1145/3592094)                                               |  ACM TOG    | [website](https://hyperhuman.deemos.com/) |

## 3D Embodied Agent
|  ID |       keywords       |    Institute   | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 1 |         RT-1         |     Google     | [RT-1: Robotics Transformer for Real-World Control at Scale](https://robotics-transformer1.github.io/assets/rt1.pdf)                                                                                                  |Arxiv|  [github](https://robotics-transformer1.github.io/) |
| 2 |         RT-2         |     Google-DeepMind     | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/pdf/2307.15818.pdf)                                                                                                  |Arxiv|  [github](https://robotics-transformer2.github.io/) |
| 3 |         SayPlan        |     QUT Centre for Robotics    | [SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://arxiv.org/pdf/2307.06135.pdf)                                                                                                  |CoRL'2023|  [github](https://sayplan.github.io/) |
| 4 |        UniHSI      |      Shanghai AI Lab     | [Unified Human-Scene Interaction via Prompted Chain-of-Contacts](https://arxiv.org/pdf/2309.07918.pdf)                                                                                                 |   Arxiv |  [github](https://github.com/OpenRobotLab/UniHSI) |
| 5 |         LLM-Planner         |     The Ohio State University    | [LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models](https://arxiv.org/pdf/2212.04088.pdf)                                                                                                  |ICCV'2023|  [github](https://github.com/OSU-NLP-Group/LLM-Planner/) |
| 6 | STEVE | ZJU & UW | [See and Think: Embodied Agent in Virtual Environment](https://arxiv.org/abs/2311.15209) | Arxiv | [github](https://github.com/rese1f/STEVE) |
| 7 |          SceneDiffuser          |      BIGAI      | [Diffusion-based Generation, Optimization, and Planning in 3D Scenes](https://arxiv.org/pdf/2301.06015)                                                           |    Arxiv  |  [github](https://github.com/scenediffuser/Scene-Diffuser) |
| 8 |          LEO          |      BIGAI      | [An Embodied Generalist Agent in 3D World](https://arxiv.org/pdf/2311.12871.pdf)                                                           |    Arxiv  |  [github](https://github.com/embodied-generalist/embodied-generalist) |


## 3D Benchmarks
|  ID |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 1 |     ScanQA     |      RIKEN AIP    | [ScanQA: 3D Question Answering for Spatial Scene Understanding](https://arxiv.org/pdf/2112.10482.pdf)                                                                                                        | CVPR'2023| [github](https://github.com/ATR-DBI/ScanQA) |
| 2 |     ScanRefer     |      TUM   | [ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/pdf/2112.10482.pdf)                                                                                                        | ECCV'2020 | [github](https://daveredrum.github.io/ScanRefer/) |
| 3 |     Scan2Cap     |      TUM    | [Scan2Cap: Context-aware Dense Captioning in RGB-D Scans](https://arxiv.org/pdf/2012.02206.pdf)                                                                                                        | CVPR'2021| [github](https://github.com/daveredrum/Scan2Cap) |
| 4 |     SQA3D     |      BIGAI    | [SQA3D: Situated Question Answering in 3D Scenes](https://arxiv.org/pdf/2210.07474.pdf)                                                                                                        | ICLR'2023| [github](https://github.com/SilongYong/SQA3D) |
| 5 |         -         |     DeepMind & UCL     | [Evaluating VLMs for Score-Based, Multi-Probe Annotation of 3D Objects](https://arxiv.org/pdf/2311.17851.pdf)                                                                                                  |Arxiv|  [github]() |



## Contributing

This is an active repository and your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for 3D LLMs, you could vote for them by adding 👍 to them.

---

If you have any questions about this opinionated list, please get in touch at xianzheng@robots.ox.ac.uk.

## Acknowledgement
This repo is inspired by [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#other-awesome-lists)

