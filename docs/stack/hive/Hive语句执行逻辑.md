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

# 关键点

## WHERE 和 HAVING 的区别

WHERE对分数前的数据进行过滤, HAVING对分组后的聚合结果进行过滤.
