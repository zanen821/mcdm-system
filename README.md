<h1>
  Multi-Criteria Decision Making System
</h1>

## 專案介紹
多準則決策的比較系統，放入 Excel 可以直接算出績效，並支援不同權重法與績效評估法自由組合比較。

## 目前支援方法
- **Weight**：
  -AHP
  -BWM(2015,2016)
  -DEMATEL
  -Entropy
  -CRITIC
  -ITARA
  -modified ITARA (I,II)
  -SECA
- **Performance**：
  -Marcos
  -promethee


## 使用指南

1. 建立 Excel 檔案，格式如下:
   
    |      |    C1   |   C2    |  C3   | ..... |
    |------|---------|---------|-------|-------|
    | type | benefit | benefit | cost  | ..... |
    | A1   | ....... | ........| ..... | ......|
    | A2   | ....... | ........| ..... | ......|
    | A3   | ....... | ........| ..... | ......|

    - **第一列(index）**：填入 'type'，用來標註每個準則是望大還是望小
      - Benefit（望大）
      - Cost（望小）
    - **第一欄**：備選方案名稱（Alternative），例如 A1、A2、A3...

2. 將 Excel 放入 'data/' 資料夾。

3. 修改 'main.py'中的檔案路徑，執行程式：

4. 輸出結果會包含各方案的績效分數（score）與排名（rank）。
   
6. 我不會寫前端 哭了


## 待完成
- [ ] 新增 Fuzzy DEMATEL 權重法
- [ ] 新增 Gray DEMATEL 權重法
- [ ] 新增 Z DEMATEL 權重法
- [ ] 新增 ANP 權重法
- [ ] 新增 CRITIC 權重法
- [ ] 新增 ITARA 權重法
- [ ] 新增 ITARA 權重法
- [ ] 新增 SECA 權重法
- [ ] 新增 FullEX 權重法
- [ ] 新增 HISA 權重法
- [ ] 新增 SAW 績效評估法
- [ ] 新增 WASPAS 績效評估法
- [ ] 新增 TOPSIS 績效評估法
- [ ] 新增 VIKOR 績效評估法
- [ ] 新增 EDAS 績效評估法
- [ ] 新增 ALWAS 績效評估法
- [ ] 新增 PROMETHEE II 績效評估法
- [ ] 新增 TODIM 績效評估法
- [ ] 新增 MABAC 績效評估法
- [ ] 新增 AROMAN 績效評估法
- [ ] 新增 RIM 績效評估法
- [ ] 新增 GRA 績效評估法
- [ ] 新增 CoCoSo 績效評估法
- [ ] 新增 MAREC 績效評估法
- [ ] 新增 DANP 績效評估法
- [ ] 建立 `compare.py`，支援多方法組合比較
