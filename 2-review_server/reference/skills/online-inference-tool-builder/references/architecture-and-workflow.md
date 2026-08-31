# 架构与任务工作流

## 1. 选择实现档位

### 轻量档

适合本机/可信内网、单用户或低并发、单项耗时在浏览器请求超时范围内的工具。

```text
Browser (static HTML/CSS/JS)
  -> Tool HTTP server
       -> model adapter -> inference service
       -> data/<task-id>/task.json + files/
```

可使用 Python 标准库或项目已有轻框架。优点是零/少依赖、复制目录即可运行；限制是任务调度依赖进程和浏览器、JSON 文件不适合多进程并发。

### 增强档

满足任一条件时优先选择：多人共享、单项耗时很长、GPU 资源需要限流、任务量大、必须取消/重试/恢复、需要权限审计。

```text
Browser -> API -> database/object storage
                -> queue -> worker -> model service
Browser <- polling/SSE/WebSocket <- task progress
```

API 创建任务后尽快返回 `202 + taskId`。worker 更新逐项状态；前端轮询或订阅进度，不持有任务执行权。

## 2. 领域契约

保持持久化结构稳定，模型供应商字段进入 `rawResponse` 或 adapter 内部。示意结构：

```json
{
  "id": "20260826_103015_ab12cd34",
  "createdAt": "2026-08-26T10:30:15+08:00",
  "updatedAt": "2026-08-26T10:30:19+08:00",
  "status": "partial_failed",
  "model": {"id": "model-x", "name": "示例模型"},
  "params": {"confidence": 0.5},
  "itemCount": 2,
  "items": [
    {
      "index": 0,
      "name": "sample-001",
      "status": "success",
      "inputs": [{"role": "A", "url": "/data/..."}],
      "result": {"type": "polygons", "regions": []},
      "error": null,
      "costMs": 920
    }
  ]
}
```

约束：

- `itemCount` 是声明总数，`items.length` 是已持久化数，二者不能混为一谈。
- 时间使用带时区 ISO 8601；耗时用单调时钟计算并显式命名单位。
- 错误至少包含可展示消息；增强版可加入稳定错误码、是否可重试和诊断 ID。
- 结果坐标必须声明单位与参考图尺寸，优先规范化为 `[0,1]`。

## 3. 状态转换

轻量档：

```text
create -> running -> completed
                  -> partial_failed
                  -> failed
```

状态汇总规则应只看已保存的逐项结果：全部成功为 `completed`；成功和失败并存为 `partial_failed`；没有成功项为 `failed`。若进程重启后发现长期 `running` 且没有执行器，应标为 `interrupted` 或提供继续/终结操作，避免永久假运行。

增强档可增加：

```text
queued -> running -> completed / partial_failed / failed
                 -> cancelling -> cancelled
```

状态更新和结果写入应处于同一事务或采用可恢复的写入顺序。

## 4. API 形状

接口命名可随项目调整，但至少覆盖：

| 能力 | 示例 |
|---|---|
| 读取安全的客户端配置 | `GET /api/config` |
| 任务列表/摘要 | `GET /api/tasks` |
| 任务详情 | `GET /api/tasks/{id}` |
| 创建任务 | `POST /api/tasks` |
| 提交单项（轻量档） | `POST /api/tasks/{id}/items` |
| 终结任务（轻量档） | `POST /api/tasks/{id}/finish` |
| 取消任务（增强档） | `POST /api/tasks/{id}/cancel` |
| 删除任务 | `DELETE /api/tasks/{id}` |

列表返回摘要，详情才返回全部逐项结果和较大字段。统一成功/错误 envelope 或使用一致 HTTP 语义，不要混用多个业务码约定。

## 5. 文件与批量配对

单组模式直接要求完整输入集合。批量模式可按相对路径同名、角色后缀（如 `_a/_b`）或用户提供清单配对，但先把规则写成纯函数并测试。

配对算法应：

1. 只接收允许类型，路径分隔符统一为 `/`。
2. 去掉文件夹选择产生的公共根目录。
3. 生成规范化键时只移除末尾角色后缀，不误改中间文本。
4. 检测同一键对应多个文件，列为冲突而非保留第一个。
5. 分别报告 A 缺 B、B 缺 A、重复键和已配对数。
6. 对配对结果按规范化键排序，使重跑顺序稳定。

浏览器 `webkitdirectory` 适合 Chromium 系工具，但不是完整标准。需要跨浏览器或超大目录时改用 zip/清单上传或服务端目录选择方案。

## 6. 持久化与恢复

轻量 JSON 存储的安全写法：任务目录固定在数据根目录下；先写同目录临时文件，刷新/关闭后用原子替换覆盖 `task.json`；每个任务写操作加锁。不要直接覆盖到一半留下损坏 JSON。

历史列表忽略单个损坏任务时，要记录诊断日志；详情读取则返回明确错误。原图、缩略图与结果使用相对 URL，不把机器绝对路径泄漏到浏览器。

删除时对用户输入的 ID使用严格字符白名单，并验证 `resolvedTask.parent == resolvedDataRoot`。删除前读取摘要用于响应；多用户系统还要验证所有权。

## 7. 前端状态组织

把状态分为：表单输入、任务列表、当前任务/项、提交进度、查看器状态。视图渲染只读取状态；所有网络请求经一个统一 API 包装器处理非 JSON、HTTP 错误和业务错误。

批量提交若由浏览器编排，逐项顺序处理通常比无限并发更适合模型服务，并能持续落盘进度。需要并发时使用小型并发池并由服务端设置最终上限。
