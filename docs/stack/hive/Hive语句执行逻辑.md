# Hive 语句的执行顺序

1. **from**
2. **join on** 和 **lateral view explode()**
3. **where**
4. **group by** (从这里开始, 可以使用`SELECT`中的别名)
5. **聚合函数**
6. **having**
7. **select**
8. **distinct**
9. **order by**
10. **limit**

上面的这些执行项可以分为两个阶段:

- **MAP阶段**
  - 执行from, 进行表的查找与加载
  - 执行where, 进行条件过滤与筛选
  - 执行select, 进行输出项的筛选
  - 执行group by分组, 描述了分组后需要计算的函数
  - map端文件合并, 每个map最终形成一个临时文件, 然后按列映射到对应的reduceReduce阶段
- **Reduce阶段**
  - 执行group by, 对map端发送过来的数据进行分组并进行计算
  - 执行having, 最后过滤列用于输出结果
  - 执行limit, 排序后进行结果输出到HDFS文件

# 关键点

## WHERE 和 HAVING 的区别

WHERE对分数前的数据进行过滤, HAVING对分组后的聚合结果进行过滤.
