# 题目描述

[712. 两个字符串的最小ASCII删除和](https://leetcode.cn/problems/minimum-ascii-delete-sum-for-two-strings/)

给定两个字符串s1, s2，找到使两个字符串相等所需删除字符的ASCII值的最小和。

示例 1:
```
输入: s1 = "sea", s2 = "eat"
输出: 231
解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。
在 "eat" 中删除 "t" 并将 116 加入总和。
结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。
```

示例 2:
```
输入: s1 = "delete", s2 = "leet"
输出: 403
解释: 在 "delete" 中删除 "dee" 字符串变成 "let"，
将 100[d]+101[e]+101[e] 加入总和。在 "leet" 中删除 "e" 将 101[e] 加入总和。
结束时，两个字符串都等于 "let"，结果即为 100+101+101+101 = 403 。
如果改为将两个字符串转换为 "lee" 或 "eet"，我们会得到 433 或 417 的结果，比答案更大。
```

注意:

- 0 < s1.length, s2.length <= 1000。
- 所有字符串中的字符ASCII值在[97, 122]之间。

# 解题思路

本题与[583. 两个字符串的删除操作](https://leetcode-cn.com/problems/delete-operation-for-two-strings/)题目类似, 操作都是通过删除两个字符串中的字符, 使得两个字符串相同. 其实等价于寻找最长公共子序列.

与583题不同的是, 本题除了要寻找到不再是最长的公共子序列, 而是这些公共子序列中两个字符串中删除的字符的ASCII编码值之和最小, 换句话说, 就是寻找所有最长公共子序列中, ASCII编码值最大的一个子序列.

因此动态规划的状态矩阵`dp[i][j]`定义为`s1`字符串的前i个字符和`s2`字符串前j个字符公共子序列对应的ASCII编码之和的最大值.

在求`dp[i][j]`时, 如果`s1[i] == s2[j]`, 说明公共子序列又可以延长一位, `dp[i][j] = dp[i - 1][j - 1] + ord(s1[i - 1])`.

如果`s1[i] != s2[j]`, 我们从`dp[i - 1][j]`和`dp[i][j - 1]`之间选取更大的值.

```python
class Solution:
    def minimumDeleteSum(self, s1: str, s2: str) -> int:
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(m):
                if s1[i] == s2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + ord(s1[i])
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        
        return sum([ord(c) for c in s1]) + sum([ord(c) for c in s2]) - dp[-1][-1] * 2
```
