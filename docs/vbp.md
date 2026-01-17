## Virtual Best Performance (VBP) metric

* The Virtual Best Performance (VBP) metric evaluates how well a portfolio of methods/algorithms could perform if, for each instance, you always picked the best-performing method among the available ones.
* It is a hypothetical upper bound: no single method can exceed VBP on the same set, because VBP assumes an oracle that selects the best method per instance.

### How it’s computed
* Given a set of instances \(i = 1,\dots,n\) and methods \(m = 1,\dots,M\), with performance measure \(p_{i,m}\) (e.g., runtime, accuracy, regret):
    * If “higher is better” (e.g., accuracy): \[\text{VBP} = \frac{1}{n} \sum_{i=1}^{n} \max_{m} \, p_{i,m}\]
    * If “lower is better” (e.g., runtime, loss): \[\text{VBP} = \frac{1}{n} \sum_{i=1}^{n} \min_{m} \, p_{i,m}\]

### Why it’s useful
* Establishes an optimistic ceiling for multi-method systems (e.g., algorithm selection, autoML, solver portfolios).
* Measures the “headroom” between the best single method and the theoretical best achievable by perfect per-instance selection.
* Helps quantify potential gains from meta-learning, ensembling, or selector policies.

### Related terms
* Virtual best solver (VBS) / oracle selector: the per-instance best method used to compute VBP.
* Single best method (SBM): the single method with the best average performance across all instances. The gap VBP − SBM indicates potential benefit of selection.

### Practical notes
* Use consistent metrics and normalization across methods (e.g., same timeout, same evaluation protocol).
* Report both VBP and SBM (and your selector’s actual performance) to contextualize results.
* For stochastic methods, compute VBP from aggregated statistics (e.g., mean over seeds) or define VBP over repeated runs carefully.