
# Awesome-LLM-3D [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity) [![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)  <a href="" target='_blank'><img src="https://visitor-badge.laobi.icu/badge?page_id=activevisionlab.llm3d&left_color=gray&right_color=blue"> </a> [![arXiv](https://img.shields.io/badge/arXiv-2405.10255-b31b1b.svg)](https://arxiv.org/abs/2405.10255)

### 📢 Survey paper available on arXiv now: **[[Paper](https://arxiv.org/pdf/2405.10255)]**

## 🏠 About
Here is a curated list of papers about 3D-Related Tasks empowered by Large Language Models (LLMs). 
It contains various tasks including 3D understanding, reasoning, generation, and embodied agents. Also, we include other Foundation Models (CLIP, SAM) for the whole picture of this area.

This is an active repository, you can watch for following the latest advances. If you find it useful, please kindly star this repo.

## 🔥 News
- [2023-12-16] [Xianzheng Ma](https://xianzhengma.github.io/) and [Yash Bhalgat](https://yashbhalgat.github.io/) curated this list and published the first version;
- [2024-01-06] [Runsen Xu](https://runsenxu.com/) added chronological information and [Xianzheng Ma](https://xianzhengma.github.io/) reorganized it in Z-A order for better following the latest advances.

## Table of Content

- [Awesome-LLM-3D](#awesome-llm-3D)
  - [3D Understanding (LLM)](#3d-understanding-via-llm)
  - [3D Understanding (other Foundation Models)](#3d-understanding-via-other-foundation-models)
  - [3D Reasoning](#3d-reasoning)
  - [3D Generation](#3d-generation)
  - [3D Embodied Agent](#3d-embodied-agent)
  - [3D Benchmarks](#3d-benchmarks)
  - [Contributing](#contributing)



## 3D Understanding via LLM

|  Date |       Keywords       |    Institute (first)   | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------:
| 2024-09-08 | MSR3D | BIGAI | [Multi-modal Situated Reasoning in 3D Scenes](https://arxiv.org/abs/2409.02389) | Axiv| [project](https://msr3d.github.io/) |
| 2024-06-07  | SpatialPIN           | Oxford                 | [SpatialPIN: Enhancing Spatial Reasoning Capabilities of Vision-Language Models through Prompting and Interacting 3D Priors](https://arxiv.org/abs/2403.13438)                         | Arxiv       | [project](https://dannymcy.github.io/zeroshot_task_hallucination/) |
| 2024-02-27 |  ShapeLLM |    XJTU  | [ShapeLLM: Universal 3D Object Understanding for Embodied Interaction](https://arxiv.org/pdf/2402.17766)                                                                                | Arxiv | [project](https://qizekun.github.io/shapellm/) |
| 2024-01-22  | SpatialVLM           | Google DeepMind        | [SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities](https://arxiv.org/abs/2401.12168)                                                                  | CVPR '24    | [project](https://spatial-vlm.github.io/) |
| 2023-12-21 |  LiDAR-LLM |    PKU  | [LiDAR-LLM: Exploring the Potential of Large Language Models for 3D LiDAR Understanding](https://arxiv.org/pdf/2312.14074.pdf)                                                                                | Arxiv | [project](https://sites.google.com/view/lidar-llm) |
| 2023-12-15 |  3DAP |    Shanghai AI Lab  | [3DAxiesPrompts: Unleashing the 3D Spatial Task Capabilities of GPT-4V](https://arxiv.org/pdf/2312.09738.pdf)                                                                                | Arxiv | [project]() |
| 2023-12-13 |  Chat-3D v2 |    ZJU | [Chat-3D v2: Bridging 3D Scene and Large Language Models with Object Identifiers](https://arxiv.org/pdf/2312.08168.pdf)                                                                                | Arxiv | [github](https://github.com/Chat-3D/Chat-3D-v2) |
| 2023-12-5 | GPT4Point | HKU | [GPT4Point: A Unified Framework for Point-Language Understanding and Generation](https://arxiv.org/pdf/2312.02980.pdf) |Arxiv |  [github](https://github.com/Pointcept/GPT4Point) |
| 2023-11-30 |         LL3DA        |     Fudan University    | [LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning](https://arxiv.org/pdf/2311.18651.pdf)                                                                                                  |Arxiv|  [github](https://github.com/Open3DA/LL3DA) |
| 2023-11-26 | ZSVG3D | CUHK(SZ) | [Visual Programming for Zero-shot Open-Vocabulary 3D Visual Grounding](https://arxiv.org/pdf/2311.15383.pdf) | Arxiv | [project](https://curryyuan.github.io/ZSVG3D/) | Arxiv | 
| 2023-11-18 |          LEO          |      BIGAI      | [An Embodied Generalist Agent in 3D World](https://arxiv.org/pdf/2311.12871.pdf)                                                           |    Arxiv  |  [github](https://github.com/embodied-generalist/embodied-generalist) |
| 2023-10-14 | JM3D-LLM | Xiamen University | [JM3D & JM3D-LLM: Elevating 3D Representation with Joint Multi-modal Cues](https://arxiv.org/pdf/2310.09503v2.pdf)                                        | ACM MM '23 |  [github](https://github.com/mr-neko/jm3d) |
| 2023-10-10 |  Uni3D |    BAAI  | [Uni3D: Exploring Unified 3D Representation at Scale](https://arxiv.org/abs/2310.06773)                                                                                | ICLR '24 | [project](https://github.com/baaivision/Uni3D) |
| 2023-9-27 |  - |    KAUST  | [Zero-Shot 3D Shape Correspondence](https://arxiv.org/abs/2306.03253)                                                                                | Siggraph Asia '23 | - |
| 2023-9-21|       LLM-Grounder       |      U-Mich      | [LLM-Grounder: Open-Vocabulary 3D Visual Grounding with Large Language Model as an Agent](https://arxiv.org/pdf/2309.12311.pdf)              |     ICRA '24     |  [github](https://github.com/sled-group/chat-with-nerf) |
| 2023-9-1 |        Point-Bind       |      CUHK     | [Point-Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following](https://arxiv.org/pdf/2309.00615.pdf)             |  Arxiv   |  [github](https://github.com/ZiyuGuo99/Point-Bind_Point-LLM) |
| 2023-8-31 |         PointLLM         |      CUHK      | [PointLLM: Empowering Large Language Models to UnderstandPoint Clouds](https://arxiv.org/pdf/2308.16911.pdf)                                                                             |   Arxiv |  [github](https://github.com/OpenRobotLab/PointLLM) |
| 2023-8-17|     Chat-3D     |      ZJU     | [Chat-3D: Data-efficiently Tuning Large Language Model for Universal Dialogue of 3D Scenes](https://arxiv.org/pdf/2308.08769v1.pdf)                                                          |  Arxiv      |  [github](https://github.com/Chat-3D/Chat-3D)|
| 2023-8-8 |         3D-VisTA          |      BIGAI      | [3D-VisTA: Pre-trained Transformer for 3D Vision and Text Alignment](https://Arxiv.org/abs/2308.04352)                                                           |    ICCV '23  | [github]() |
| 2023-7-24 |     3D-LLM     |      UCLA    | [3D-LLM: Injecting the 3D World into Large Language Models](https://arxiv.org/pdf/2307.12981.pdf)                                                                                                                      |   NeurIPS '23|  [github](https://github.com/UMass-Foundation-Model/3D-LLM) |
| 2023-3-29 |       ViewRefer       |      CUHK      | [ViewRefer: Grasp the Multi-view Knowledge for 3D Visual Grounding](https://arxiv.org/pdf/2303.16894.pdf)                                                                                               |ICCV '23 |[github](https://github.com/Ivan-Tang-3D/ViewRefer3D) |
| 2022-9-12 |        -        |      MIT      | [Leveraging Large (Visual) Language Models for Robot 3D Scene Understanding](https://arxiv.org/pdf/2209.05629.pdf)                                                                      |Arxiv|  [github](https://github.com/MIT-SPARK/llm_scene_understanding) |


## 3D Understanding via other Foundation Models
|  ID |       keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 2024-04-07 |  Any2Point |    Shanghai AI Lab  | [Any2Point: Empowering Any-modality Large Models for Efficient 3D Understanding](https://arxiv.org/pdf/2404.07989) | ECCV 2024 | [github](https://github.com/Ivan-Tang-3D/Any2Point) |
| 2024-03-16 |  N2F2 |    Oxford-VGG  | [N2F2: Hierarchical Scene Understanding with Nested Neural Feature Fields](https://arxiv.org/pdf/2403.10997.pdf) | Arxiv | - |
| 2023-12-17 |  SAI3D |    PKU  | [SAI3D: Segment Any Instance in 3D Scenes](https://arxiv.org/pdf/2312.11557.pdf)                                                                                | Arxiv | [project](https://yd-yin.github.io/SAI3D) |
| 2023-12-17 |  Open3DIS |    VinAI  | [Open3DIS: Open-vocabulary 3D Instance Segmentation with 2D Mask Guidance](https://arxiv.org/pdf/2312.10671.pdf)                                                                                | Arxiv | [project](https://open3dis.github.io/) |
| 2023-11-6 |  OVIR-3D |    Rutgers University  | [OVIR-3D: Open-Vocabulary 3D Instance Retrieval Without Training on 3D Data](https://arxiv.org/pdf/2311.02873.pdf) | CoRL '23 | [github](https://github.com/shiyoung77/OVIR-3D/) |
| 2023-10-29|  OpenMask3D |    ETH  | [OpenMask3D: Open-Vocabulary 3D Instance Segmentation](https://openmask3d.github.io/static/pdf/openmask3d.pdf)                                                                                | NeurIPS '23 | [project](https://openmask3d.github.io/) |
| 2023-10-5 |     Open-Fusion     |      -     | [Open-Fusion: Real-time Open-Vocabulary 3D Mapping and Queryable Scene Representation](https://arxiv.org/pdf/2310.03923.pdf)                                                                            |Arxiv|  [github](https://github.com/UARK-AICV/OpenFusion) |
| 2023-9-22 |  OV-3DDet |    HKUST  | [CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection](https://arxiv.org/pdf/2310.02960.pdf)                                                                                | NeurIPS '23 | [github](https://github.com/yangcaoai/CoDA_NeurIPS2023) |
| 2023-9-19 | LAMP |      -      | [From Language to 3D Worlds: Adapting Language Model for Point Cloud Perception](https://openreview.net/forum?id=H49g8rRIiF)                                                              |    OpenReview     | - |
| 2023-9-15 |  OpenNerf |    -    | [OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views](https://openreview.net/pdf?id=SgjAojPKb3)                                                                                | OpenReview | [github]() |
| 2023-9-1|  OpenIns3D |    Cambridge  | [OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance Segmentation](https://arxiv.org/pdf/2309.00616.pdf)                                                                                | Arxiv | [project](https://zheninghuang.github.io/OpenIns3D/) |
| 2023-6-7 |         Contrastive Lift         |     Oxford-VGG     | [Contrastive Lift: 3D Object Instance Segmentation by Slow-Fast Contrastive Fusion](https://arxiv.org/pdf/2306.04633.pdf)                                                                                        |   NeurIPS '23| [github](https://github.com/yashbhalgat/Contrastive-Lift) |
| 2023-6-4 |  Multi-CLIP |    ETH  | [Multi-CLIP: Contrastive Vision-Language Pre-training for Question Answering tasks in 3D Scenes](https://arxiv.org/pdf/2306.02329.pdf)                                                                                | Arxiv | - |
| 2023-5-23 |  3D-OVS |    NTU  | [Weakly Supervised 3D Open-vocabulary Segmentation](https://arxiv.org/pdf/2305.14093.pdf)                                                                                | NeurIPS '23 | [github](https://github.com/Kunhao-Liu/3D-OVS) |
| 2023-5-21 |  VL-Fields |    University of Edinburgh  | [VL-Fields: Towards Language-Grounded Neural Implicit Spatial Representations](https://arxiv.org/pdf/2305.12427.pdf)                                                                                | ICRA '23 | [project](https://tsagkas.github.io/vl-fields/)  |
| 2023-5-8 |  CLIP-FO3D |    Tsinghua University  | [CLIP-FO3D: Learning Free Open-world 3D Scene Representations from 2D Dense CLIP](https://arxiv.org/pdf/2303.04748.pdf)                                                                                | ICCVW '23 | - |
| 2023-4-12 |  3D-VQA |    ETH  | [CLIP-Guided Vision-Language Pre-training for Question Answering in 3D Scenes](https://arxiv.org/pdf/2304.06061.pdf)                                                                                | CVPRW '23 | [github](https://github.com/AlexDelitzas/3D-VQA) |
| 2023-4-3 |  RegionPLC |    HKU | [RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding](https://arxiv.org/pdf/2304.00962.pdf)                                                                                | Arxiv | [project](https://jihanyang.github.io/projects/RegionPLC) |
| 2023-3-20 |        CG3D        |      JHU      | [CLIP goes 3D: Leveraging Prompt Tuning for Language Grounded 3D Recognition](https://arxiv.org/pdf/2303.11313.pdf)                                                                                                 |Arxiv|  [github](https://github.com/deeptibhegde/CLIP-goes-3D) |
| 2023-3-16 | LERF |     UC Berkeley     | [LERF: Language Embedded Radiance Fields](https://arxiv.org/pdf/2303.09553.pdf)                                                   | ICCV '23   | [github](https://github.com/kerrj/lerf) |
| 2023-2-14 |  ConceptFusion |    MIT  | [ConceptFusion: Open-set Multimodal 3D Mapping](https://arxiv.org/pdf/2302.07241.pdf)                                                                                | RSS '23 | [project](https://concept-fusion.github.io/) |
| 2023-1-12 |         CLIP2Scene         |      HKU      | [CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP](https://arxiv.org/pdf/2301.04926.pdf)                                                                                        |    CVPR '23 | [github](https://github.com/runnanchen/CLIP2Scene) |
| 2022-12-1 |         UniT3D         |      TUM     | [UniT3D: A Unified Transformer for 3D Dense Captioning and Visual Grounding](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_UniT3D_A_Unified_Transformer_for_3D_Dense_Captioning_and_Visual_ICCV_2023_paper.pdf)                                                                          |   ICCV '23| [github]() |
| 2022-11-29 |        PLA        |     HKU    | [PLA: Language-Driven Open-Vocabulary 3D Scene Understanding](https://arxiv.org/pdf/2211.16312.pdf)                                                                 |CVPR '23|  [github](https://github.com/CVMI-Lab/PLA) |
| 2022-11-28 |       OpenScene       |      ETHz      | [OpenScene: 3D Scene Understanding with Open Vocabularies](https://arxiv.org/pdf/2211.15654.pdf)                                                             |   CVPR '23  | [github](https://github.com/pengsongyou/openscene) |
| 2022-10-11 |  CLIP-Fields |    NYU  | [CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory](https://arxiv.org/pdf/2210.05663.pdf)                                                                                | Arxiv | [project](https://mahis.life/clip-fields/) |
| 2022-7-23 |  Semantic Abstraction |    Columbia  | [Semantic Abstraction: Open-World 3D Scene Understanding from 2D Vision-Language Models](https://arxiv.org/pdf/2207.11514.pdf)                                                                                | CoRL '22 | [project](https://semantic-abstraction.cs.columbia.edu/) |
| 2022-4-26 |   ScanNet200 |    TUM  | [Language-Grounded Indoor 3D Semantic Segmentation in the Wild](https://arxiv.org/pdf/2204.07761.pdf)                                                                                | ECCV '22 | [project](https://rozdavid.github.io/scannet200) |




## 3D Reasoning
|  Date |       keywords       |    Institute (first)    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 2023-5-20|       3D-CLR      |      UCLA     | [3D Concept Learning and Reasoning from Multi-View Images](https://arxiv.org/pdf/2303.11327.pdf)                                                 |   CVPR '23  | [github](https://github.com/evelinehong/3D-CLR-Official) |
| - |         Transcribe3D         |     TTI, Chicago     | [Transcribe3D: Grounding LLMs Using Transcribed Information for 3D Referential Reasoning with Self-Corrected Finetuning](https://openreview.net/pdf?id=7j3sdUZMTF)                                                                                                  |CoRL '23|  [github]() |


## 3D Generation
|  Date |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 2023-11-29 |         ShapeGPT         |     Fudan University     | [ShapeGPT: 3D Shape Generation with A Unified Multi-modal Language Model](https://arxiv.org/pdf/2311.17618.pdf)                                                                                                  |Arxiv|  [github](https://github.com/OpenShapeLab/ShapeGPT) |                                                                                              | Arxiv  |  [github]() |
| 2023-11-27|         MeshGPT         |     TUM     | [MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers](https://arxiv.org/pdf/2311.15475.pdf)                                                                                                  |Arxiv |  [project](https://nihalsid.github.io/mesh-gpt/) |
| 2023-10-19 |         3D-GPT        |     ANU   | [3D-GPT: Procedural 3D Modeling with Large Language Models](https://arxiv.org/pdf/2310.12945.pdf)                                                                                                   |Arxiv|  [github](https://dreamllm.github.io/) |
| 2023-9-21 |         LLMR         |     MIT     | [LLMR: Real-time Prompting of Interactive Worlds using Large Language Models](https://arxiv.org/pdf/2309.12276.pdf)                                                                                                  |Arxiv|  [github]() |
| 2023-9-20 |         DreamLLM         |     MEGVII    | [DreamLLM: Synergistic Multimodal Comprehension and Creation](https://arxiv.org/pdf/2309.11499.pdf) | Arxiv | [github](https://github.com/RunpeiDong/DreamLLM)
| 2023-4-1 |      ChatAvatar      |       Deemos Tech            | [DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance](https://dl.acm.org/doi/abs/10.1145/3592094)                                               |  ACM TOG    | [website](https://hyperhuman.deemos.com/) |

## 3D Embodied Agent
|  Date |       keywords       |    Institute   | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 2023-11-27 | Dobb-E | NYU | [On Bringing Robots Home](https://arxiv.org/pdf/2311.16098.pdf)        |    Arxiv  |  [github](https://github.com/notmahi/dobb-e) |
| 2023-11-26 | STEVE | ZJU | [See and Think: Embodied Agent in Virtual Environment](https://arxiv.org/abs/2311.15209) | Arxiv | [github](https://github.com/rese1f/STEVE) |
| 2023-11-18 | LEO  |   BIGAI  | [An Embodied Generalist Agent in 3D World](https://arxiv.org/pdf/2311.12871.pdf)   |    Arxiv  |  [github](https://github.com/embodied-generalist/embodied-generalist) |
| 2023-9-14 |        UniHSI      |      Shanghai AI Lab     | [Unified Human-Scene Interaction via Prompted Chain-of-Contacts](https://arxiv.org/pdf/2309.07918.pdf)                                                                                                 |   Arxiv |  [github](https://github.com/OpenRobotLab/UniHSI) |
| 2023-7-28 |         RT-2         |     Google-DeepMind     | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/pdf/2307.15818.pdf)                                                                                                  |Arxiv|  [github](https://robotics-transformer2.github.io/) |
| 2023-7-12 |         SayPlan        |     QUT Centre for Robotics    | [SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning](https://arxiv.org/pdf/2307.06135.pdf)                                                                                                  |CoRL '23|  [github](https://sayplan.github.io/) |
| 2023-7-12 |          VoxPoser          |      Stanford      | [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models](https://voxposer.github.io/voxposer.pdf)                                                           |    Arxiv  |  [github](https://github.com/huangwl18/VoxPoser) |
| 2022-12-13|         RT-1         |     Google     | [RT-1: Robotics Transformer for Real-World Control at Scale](https://robotics-transformer1.github.io/assets/rt1.pdf)                                                                                                  |Arxiv|  [github](https://robotics-transformer1.github.io/) |
| 2022-12-8 |         LLM-Planner         |     The Ohio State University    | [LLM-Planner: Few-Shot Grounded Planning for Embodied Agents with Large Language Models](https://arxiv.org/pdf/2212.04088.pdf)                                                                                                  |ICCV '23|  [github](https://github.com/OSU-NLP-Group/LLM-Planner/) |
| 2022-10-11 |          CLIP-Fields          |      NYU, Meta      | [CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory](https://arxiv.org/pdf/2210.05663.pdf)                                                           |    RSS '23  |  [github](https://github.com/notmahi/clip-fields) |


## 3D Benchmarks
|  Date |       keywords       |    Institute    | Paper                                                                                                                                                                               | Publication | Others |
| :-----: | :------------------: | :--------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------: | :---------: 
| 2024-09-08 | MSQA / MSNN | BIGAI | [Multi-modal Situated Reasoning in 3D Scenes](https://arxiv.org/abs/2409.02389) | Axiv| [project](https://msr3d.github.io/) |
| 2024-06-10 | 3D-GRAND / 3D-POPE | UMich | [3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination](https://arxiv.org/pdf/2406.05132.pdf) | Arxiv | [project](https://3d-grand.github.io) |
| 2024-1-18 | SceneVerse | BIGAI | [SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](https://arxiv.org/pdf/2401.09340.pdf) | Arxiv | [github](https://github.com/scene-verse/sceneverse) |
| 2023-12-26 | EmbodiedScan | Shanghai AI Lab | [EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite Towards Embodied AI](https://arxiv.org/pdf/2312.16170.pdf) | Arxiv | [github](https://github.com/OpenRobotLab/EmbodiedScan) |
| 2023-12-17 |         M3DBench        |     Fudan University     | [M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts](https://arxiv.org/abs/2312.10763)                                                                                                  |Arxiv|  [github](https://github.com/OpenM3D/M3DBench) |
| 2023-11-29 |         -         |     DeepMind  | [Evaluating VLMs for Score-Based, Multi-Probe Annotation of 3D Objects](https://arxiv.org/pdf/2311.17851.pdf)                                                                                                  |Arxiv|  [github]() |
| 2022-10-14 |     SQA3D     |      BIGAI    | [SQA3D: Situated Question Answering in 3D Scenes](https://arxiv.org/pdf/2210.07474.pdf)                                                                                                        | ICLR '23| [github](https://github.com/SilongYong/SQA3D) |
| 2021-12-20|     ScanQA     |      RIKEN AIP    | [ScanQA: 3D Question Answering for Spatial Scene Understanding](https://arxiv.org/pdf/2112.10482.pdf)                                                                                                        | CVPR '23| [github](https://github.com/ATR-DBI/ScanQA) |
| 2020-12-3 |     Scan2Cap     |      TUM    | [Scan2Cap: Context-aware Dense Captioning in RGB-D Scans](https://arxiv.org/pdf/2012.02206.pdf)                                                                                                        | CVPR '21| [github](https://github.com/daveredrum/Scan2Cap) |
| 2020-8-23 | ReferIt3D | Stanford | [ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification in Real-World Scenes](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460409.pdf) | ECCV '20 | [github](https://github.com/referit3d/referit3d) 
| 2019-12-18 |     ScanRefer     |      TUM   | [ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/pdf/2112.10482.pdf)                                                                                                        | ECCV '20 | [github](https://daveredrum.github.io/ScanRefer/) |

## Contributing

your contributions are always welcome!

I will keep some pull requests open if I'm not sure if they are awesome for 3D LLMs, you could vote for them by adding 👍 to them.

---

If you have any questions about this opinionated list, please get in touch at xianzheng@robots.ox.ac.uk or Wechat ID: mxz1997112.

## Acknowledgement
This repo is inspired by [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM?tab=readme-ov-file#other-awesome-lists)

