<h1>
  Multi-Criteria Decision Making System
</h1>

## 專案介紹
多準則決策的比較系統，放入 Excel 可以直接算出績效，並支援不同權重法與績效評估法自由組合比較。

## 目前支援方法
- **Weight**：Entropy
- **Performance**：Marcos

## 使用指南

1. 建立 Excel 檔案，格式如下：

    |      |    C1   |   C2    |  C3   | ..... |
    | type | benefit | benefit | cost  | ..... |
    | A1   | ....... | ........| ..... | ......|
    | A2   | ....... | ........| ..... | ......|
    | A3   | ....... | ........| ..... | ......|

    - **第一列(index）**：填入 'type'，用來標註每個準則是望大還是望小
      - Benefit（望大）
      - Cost（望小）
    - **第一欄**：方案名稱（Alternative），例如 A1、A2、A3...

2. 將 Excel 放入 'data/' 資料夾。

3. 修改 'main.py'中的檔案路徑，執行程式：

4. 輸出結果會包含各方案的績效分數（score）與排名（rank）。

5.我不會寫前端 哭了
