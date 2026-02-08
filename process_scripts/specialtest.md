HNEI

---

## 论文中关于RPT测试的描述（总结）

1. **RPT测试的位置和作用**：
    - 在对电池进行初步调理（conditioning procedures）后，作者对每个电池进行了一个RPT测试。
    - 这个RPT用于测量一些关键性能指标，如热力学容量、容量比、伪OCV曲线（pseudo-OCV curve）、内阻（internal series resistance）、倍率性能（rate capability）。
2. **RPT测试的具体步骤和频率**：
    - 初始RPT流程包括在推荐的电压窗口 (3.0–4.3 V) 下，连续对每个电池进行对称恒流充放电循环，倍率分别为 C/25、C/5 和 C/1。
    - 选出的15个样品电池进行循环老化实验后，每周期（每100个循环）进行一次简化RPT：C/25倍率的放电-充电-放电组合，用于周期性表征。
    - 简化RPT主要包括连续的低倍率（C/25）放电-充电-放电，用于跟踪容量等关键参数变化。
    - 每25个循环，还安排了“扩展循环”，倍率同样为C/2充电、1.5 C放电，但增加了静置（rest）阶段。
3. **处理数据集需注意的点**：
    - 使用他们的数据集时，要特别留意不同倍率下的测试数据（C/25, C/5, C/1, C/2, 1.5 C），以及每个测试周期（如每100个循环后的RPT）。

---

## 论文关于RPT测试的**原文**引用

> Third, a reference performance test (RPT) was performed on every cell. This initial RPT schedule consisted of successive symmetrical constant-current cycles at C/25, C/5, and C/1 rates, within the recommended voltage window. Using the data collected as part of the RPT, some key cell performance metrics were derived for each cell in the batch.
> 

> Cycle-aging was carried out in 10 sequences of 100 cycles. After each sequence, a simplified RPT was carried out. These simplified RPTs were the basis for our periodic characterization protocol and consisted of a C/25 discharge followed by a C/25 charge and a final C/25 discharge.
> 

> In addition to these RPTs, each cell was subjected to an extended cycle every 25 cycles of the cycle-aging schedule. These extended cycles were still carried out at a C/2 charge rate and the discharge rate was 1.5 C but a total of three 4-h rest periods were imposed before the charge, after the charge, and after the discharge.
> 

---

## 数据处理建议

- **提取RPT测试数据（C/25, C/5, C/1倍率）作为电池初始性能基准**。
- **每100个循环后的简化RPT结果（C/25倍率）用于跟踪电池性能的周期性变化**。
- **扩展循环（每25周期，含静置阶段）数据可以用于分析电池在极限工况下的恢复与极化行为**。

SNL

这篇论文确实提到了对电池进行EIS（电化学阻抗谱，Electrochemical Impedance Spectroscopy）测试：

- 论文原文如下：“A round of cycling for each cell varied from 125 to 1000 cycles, depending on the rate of degradation at the specific test conditions. The cycle count for a round was halved if a cell experienced over 5% capacity loss in the previous round. These adjustments were intended to offer enough granulation in the capacity decline curve to enable observation of any changes in mechanism, while still maintaining a reasonable check-up frequency during a nearly threeyear study. Electrochemical impedance spectroscopy (EIS) was completed at intervals of approximately 3% capacity loss, and these results will be expanded upon in a future publication. For the purpose of this publication, the study was co…”
- 论文明确说明在每3%电池容量损失时进行了EIS测试，以观察阻抗变化和退化机理变化。因此，在数据处理时，你可以根据每3%容量下降的点来筛选和分析相关EIS测试数据。
- 然而，此篇论文并未详细描述EIS测试的具体频率范围，频率细节会在后续研究出版物中进一步展开。如果你要对他们的数据集做处理，只能确认EIS数据点的选择标准是每约3%容量损失间隔。具体频率参数需要参考数据集说明或等待后续论文。原文并未披露频率设定。

**数据处理建议：**

- 要处理这个数据集时，应关注每次容量损失达到3%后记录的EIS结果。这些点反映了不同退化阶段的阻抗状态。
- 如果需要EIS的频率范围，建议查阅数据集的附录或相关后续文章，或直接在batteryarchive.org查询原始数据文件和补充说明。

原文引用如下：

> “A round of cycling for each cell varied from 125 to 1000 cycles, depending on the rate of degradation at the specific test conditions... Electrochemical impedance spectroscopy (EIS) was completed at intervals of approximately 3% capacity loss, and these results will be expanded upon in a future publication. For the purpose of this publication, the study was co…”libdb.csu
> 

**总结：**

本论文提到按照每3%容量损失周期进行EIS测试，但未披露具体频率，建议按3%容量损失间隔处理EIS数据，频率参数需进一步查证或参考数据说明。[libdb.csu](https://libdb.csu.edu.cn/https/vpn/495/NFYYA65DNFTX6Z5FF3VX86BPN73GH/article/10.1149/1945-7111/abae37/pdf)

1. https://libdb.csu.edu.cn/https/vpn/495/NFYYA65DNFTX6Z5FF3VX86BPN73GH/article/10.1149/1945-7111/abae37/pdf

MICH_EXP

这篇论文明确介绍了他们对锂离子电池进行了RPT（Reference Performance Test，参考性能测试），并且频率、方法在原文中有详细描述。

**1. RPT测试相关原文：**

> Reference performance test (RPT).—Before the start of the aging tests initial RPTs were done for all the cells. The subsequent RPTs were performed after a certain number of cycles corresponding to an expected 5% capacity loss for cycling aging tests. For RPT the cells were brought back to the room temperature (25 °C) and held at rest for 3 h to ensure thermal equilibrium. The RPTs are as the following:
> 
> 1. A C/20 charge-discharge cycle which consists of an initial C/5 discharge until reaching 3.0 V, followed by a constant voltage (CV) phase at 3.0 V until (|I| < C/50) and 1 hour rest to ensure the cell is fully discharged. Then a C/20 charge until reaching 4.2 V, followed by a constant voltage (CV) phase at 4.2 V until (I < C/50) and 1 hour rest. Then a C/20 discharge until reaching 3.0 V. The C/20 data is used to approximate the open circuit response of the battery.
> 2. Hybrid pulse power characterization (HPPC) and electrochemical impedance spectroscopy (EIS) measurements at 10% SOC intervals. First the cells are charged using C/2 CC until 4.2 V, followed by a CV at 4.2 V until (I < C/50) and 1/2 hour rest. Then a C/2 CC discharge for an equivalent of 10% SOC discharge, where the discharge time was adjusted based on the prior capacity measurement (C/20) test. Followed by a 1/2 hour rest. Then the HPPC profile was done which consists of a 1C CC discharge for 10 s, a 10 min rest, a 1C CC discharge for 10 s. Followed by a 10 min rest. Then the EIS measurement was done over frequency range of 10 mHz–10 kHz. The above steps were repeated until the end of discharge 3.0 V was reached.

**简要整理：**

- 测试频率：
    - RPT包含极慢速充放电（C/20，即1/20小时率），这是近似开路响应的频率，也是数据处理时主要关注的慢速测试。
    - EIS测试频率为10 mHz–10 kHz（即10毫赫兹到10千赫兹）。
    - HPPC过程中，有10秒1C放电脉冲（即以电池额定电流的倍率快速释放），间隔10分钟休息。
- RPT测试流程：
    - 在衰老实验前以及每经历一段时间（按电池容量衰减到5%为节点），都会做一次RPT。
    - 每次RPT是在室温25°C下完成，并在测试前静置3小时保证热平衡。
    - 步骤包括：C/20充放电循环（非常慢的充放电）、C/2充电/放电（较快的充放电）、HPPC与EIS测试。
    - EIS的扫描频率范围是10 mHz–10 kHz。

**2. RPT数据处理建议：**

- 如果你需要用他们的数据集做处理，重点关注C/20的数据（用于容量分析和开路电压分析）

**3. 论文相关原文摘录（英文部分已展示）**

如需中文摘要或数据处理思路，可以补充说明你的后续需求。

1. https://libdb.csu.edu.cn/https/vpn/495/NFYYA65DNFTX6Z5FF3VX86BPN73GH/article/10.1149/1945-7111/ac2d3e/pdf

MICH_EXP

该论文详细介绍了他们对电池进行的RPT（Reference Performance Test，参考性能测试），并对测试频率和内容进行了说明。以下是论文原文关于RPT测试的描述和频率相关细节总结：

**1. RPT测试相关原文：**

在论文第3页（RESULTS AND DISCUSSION，Fast formation experimental design），描述如下：

> Reference performance tests (RPTs)41 were inserted throughout the cycle life test, which includes slow (C/20) charge and discharge curves as well as a hybrid pulse power characterization (HPPC) sequence42 used to extract the cell internal resistance as a function of SOC.
> 
> 
> ... The cycling profile was identical for all cells: 1 C charge to 4.2 V with a constant voltage (CV) hold to 10 mA and 1 C discharge to 3.0 V.
> 

**2. 具体测试内容和频率说明（整理总结）：**

- RPT测试包含：
    - 慢速充电与放电曲线（C/20倍率）
    - HPPC测试序列 —— 用于提取电池在不同SOC（荷电状态）下的内阻
- RPT的插入频率为“throughout the cycle life test”，即在整个循环寿命测试过程中插入RPT测试。例如，在图1描述的循环容量曲线中，曲线中的间断处即为嵌入的RPT循环。
- HPPC测试具体方法描述（第6页）：
    - 在RPT过程中，在不同SOC下对电池施加10秒、1C倍率的放电脉冲，记录由欧姆定律计算出的电池内阻。

> During this test, a series of 10-s, 1 C discharge pulses were applied to the cell at varying SOCs, and the resistance is calculated using Ohm’s law (Figure S8).
> 
- RPT测点的SOC范围为低SOC（如4%、5%、8%、10%），以及高SOC（如90%），论文重点分析了低SOC（如5%）下的阻值变化。
- 原文表格Table 1总结了RPT内阻测量条件：如“R10s,5%SOC (RLS)”表示在5% SOC下10秒脉冲测得的阻值。

**3. 你的数据处理建议：**

- 你需要关注每个RPT测试周期插入的慢充慢放电数据和脉冲内阻数据，尤其是低SOC（如4-10%）下的10秒1C脉冲阻值。

**4. 论文原文引用（选取与RPT相关的主要原文段落）：**

- “Reference performance tests (RPTs)41 were inserted throughout the cycle life test, which includes slow (C/20) charge and discharge curves as well as a hybrid pulse power characterization (HPPC) sequence42 used to extract the cell internal resistance as a function of SOC.”[cell](https://www.cell.com/action/showPdf?pii=S2542-4351%2821%2900443-8)
- “During this test, a series of 10-s, 1 C discharge pulses were applied to the cell at varying SOCs, and the resistance is calculated using Ohm’s law (Figure S8).”[cell](https://www.cell.com/action/showPdf?pii=S2542-4351%2821%2900443-8)
- “To study the robustness of the low-SOC resistance signal, we varied the SOC setpoint between 4% and 10% and also computed the resistance under 1 and 5-s pulse durations. In all cases, the resistance metric provided a high degree of contrast between the two different formation protocols (Figures S10 and S11). The lowest SOC measured in our dataset was 4% SOC.
    
    The remainder of the paper will focus on the resistance measured at 5% SOC and with a 10-s pulse duration. From hereon, this metric will be referred to as the ‘low-SOC resistance,’ RLS.”[cell](https://www.cell.com/action/showPdf?pii=S2542-4351%2821%2900443-8)
    

如果你要处理他们的数据集，重点需提取每个样本在RPT阶段不同SOC下（特别是低SOC 4%/5%/8%/10%）的10秒脉冲阻值（RLS），这些数据即为他们寿命预测的核心特征。

如需其他细节可以再补充。

1. https://www.cell.com/action/showPdf?pii=S2542-4351%2821%2900443-8

RWTH

好的，只提供Characterization test测试的频率和原文：

**Characterization test测试的频率：**

- 在电池的BOL（Beginning Of Life，寿命开始）时进行了一次。
- 之后在循环测试之间定期重复进行，直到电池达到EOL（End Of Life，寿命结束）。
- 平均每块电池进行了17次Characterization test。
- 平均每进行大约160个充放电循环后，会进行一次Characterization test测试。

**论文原文：**

1. **关于Characterization test测试的组成和周期性：**
    
    > The cells underwent cycling ageing, consisting of characterization tests and cycling tests.
    (第二页，"2. Ageing test and dataset" 段落)
    > 
    
    > One characterization test was undertaken at the BOL for each cell and then repeated periodically between cycling tests until the cells reached their EOL.
    (第七页，"Ageing experiment" 段落)
    > 
2. **关于Characterization test的平均次数：**
    
    > There were, on average, 17 characterization tests in each cell's life.
    (第七页，"Ageing experiment" 段落)
    > 
3. **关于循环测试的平均次数（用于推断Characterization test的间隔）：**
    
    > There were, on average, 160 charging-discharging cycles in each cycling round.
    (第七页，"Ageing experiment" 段落)
    > 

TONGJI

论文原文中关于电池 ESI（交流电化学阻抗）测试的相关频率、周期及影响说明如下：

**1. 测试频率范围与具体参数：**

- NCA 和 NCM 电池在满电状态下，采用频率范围为 **10 kHz 到 50 mHz**（每十进制频率段采集 20 个数据点），激励幅度为 20 mV。
- NCM+NCA 电池在满电状态下采用频率范围 **10 kHz 到 0.01 Hz**（每十进制频率段采集 6 个数据点），激励幅度为 250 mA。

**2. 测试周期/间隔：**

- NCA 电池每 **25圈**做一次阻抗测试。
- NCM 电池每 **50圈**做一次阻抗测试。
- NCM+NCA 电池满电状态下每 **50圈**做一次阻抗测试。

**3. ESI测试前的休息时间：**

- NCA 和 NCM 电池在开路电压下静置 **30分钟**后进行阻抗测试。
- NCM+NCA 电池在开路电压下静置 **60分钟**后进行阻抗测试。

**原文出处：**

> “For partially charged/discharged NCA and NCM cells, the electrochemical impedance is measured in the fully charged state using a frequency range of 10 kHz to 50 mHz (20 data points per decade of frequency) and a potential amplitude of 20 mV. 30 min are set at the open circuit voltage before the electrochemical impedance tests. The electrochemical impedance is tested every 25 cycles for the NCA battery and every 50 cycles for the NCM battery. For the NCM + NCA battery, the electrochemical impedance is conducted every 50 cycles at full charge in a range of 10 kHz to 0.01 Hz (6 data points per decade of frequency) with a sinusoidal amplitude of 250 mA. 60 min are set at the open circuit voltage before the electrochemical impedance tests.”
> 

**建议数据处理方法：**

由于论文说明 ESI 测试周期明确，每次 ESI测试对应的圈数（NCA每25圈、NCM每50圈、NCM+NCA每50圈），而测试会对循环中的 SOH（健康状态估算）有影响，因此强烈建议在处理数据集时将所有 ESI测试圈次的数据**剔除**，避免模型误判或因外部扰动导致判据偏移，从而提升你数据驱动模型的泛化能力。

如需进一步圈次索引定位，请根据论文的数据说明对原始数据的循环计数进行分段，剔除相关测试点即可。

1. file:///C:/Users/djt/Downloads/s41467-022-29837-w.pdf