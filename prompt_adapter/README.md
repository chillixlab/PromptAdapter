## PromptAdapterの開発

### STEP1 トップモデル$M^*$の選定
1. 用意したAllganizeデータを用いて複数のLLMモデル $\mathbb{M_i} = \{ M_1,M_2,...M_I \}$に対し、RAGを用いた回答生成を実行する
   - RAGの前処理はKNSや自前実装でも問わない。実験の設定として埋め込みモデル、チャンクサイズ、オーバーラップサイズが管理出来れいればよい。
  
2. 得られた$M_i$に対し、RAGASによる評価を実行
   - RAGAS : https://arxiv.org/abs/2309.15217
   - ライブラリ：https://docs.ragas.io/en/stable/
   - 評価指標は以下 : 検索と推論の両方で評価する
     - ContextRecall（コンテキストの再現率）
       - 引数：Ground Truth Answer, Context
     - ContextPrecision（コンテキストの精度）
       - 引数：Query, Context
     - Faithfulness（忠実度）
       - 引数：Response, Context
     - Answer Relevancy（回答の関連性）
       - 引数：Query, Response
   - スコア範囲は[0.0~1.0]で高いほど良い。
   - 得られる結果イメージ   
        ```
        model        recall  precision  faithfulness  relevancy
        gpt-4o       0.90    0.80       0.88          0.91
        gpt-4.1      0.87    0.85       0.92          0.89
        gemini-2.5   0.92    0.70       0.80          0.90
        ...
        ```

3. パレート最適集合を作成し、その後トップモデル$M^*$を決定する
   - 上記の指標はモデルごとに各指標のトレードオフな状態が発生すると考えられる。（この値はgemini強いけど、こっちはgpt強いなど）
   - パレート最適集合$\mathbb{M^{pareto}}$ をいったん作り、下位モデルを振るい落とす。（実装は自前で作る）
   - その後、$\mathbb{M^{pareto}}$の中でContext Recallが最も高いモデルを$M^*$として決定
     - なぜContext Recallなのか？
     - →RAGを組み込んだ実システムを考えるうえで、回答生成の為に必要な情報がデータソースから獲得できないことが一番課題になる。そこで今回はContext Recallが良い＝いい検索が出来ているという前提を置いてトップモデルを決定する
  
## STEP2 トップモデル$M^*$をターゲットにしたAdapterの設計
- wip


