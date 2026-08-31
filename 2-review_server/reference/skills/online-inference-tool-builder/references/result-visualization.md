# 推理结果可视化

## 按结果类型选 renderer

- 分类、评分：标签、分数、阈值和原始尺寸摘要。
- 文本/OCR：结构化字段 + 原图定位；长文本支持复制与导出。
- 检测框：SVG 或 Canvas 均可；区域较少且需 DOM 交互时 SVG 更方便。
- 多边形、掩膜、海量点：Canvas 更高效；复杂交互可使用成熟图形库，但先确认依赖是否符合部署条件。
- A/B、时序或多模态对应输入：共享归一化视图状态，实现同步缩放、平移、指针和选中区域。

## 坐标约定

领域结果优先保存归一化坐标：

```text
nx = source_x / source_width
ny = source_y / source_height
```

显示时，若图片左上角为 `(offsetX, offsetY)`，缩放为 `scale`：

```text
screen_x = offsetX + nx * naturalWidth  * scale
screen_y = offsetY + ny * naturalHeight * scale
```

使用图片 `naturalWidth/naturalHeight`，不要用 CSS 尺寸代替源尺寸。若模型基于裁剪、letterbox 或旋转后的图推理，adapter 必须先做逆变换。

## 同步 A/B 查看器

共享状态使用：

```js
view = { zoom: 1, centerX: 0.5, centerY: 0.5 }
pointer = { nx: 0.5, ny: 0.5 }
```

每个画布根据自身图片自然尺寸计算 `fit = min(viewWidth/imageWidth, viewHeight/imageHeight)`，最终 `scale = fit * zoom`。共享的是归一化中心与指针，而不是像素偏移，这样不同分辨率的 A/B 图仍保持同一相对位置。

以指针为中心缩放时，应在缩放前取得指针的归一化位置，再更新中心，使该位置缩放后仍落在原屏幕点。平移后限制中心范围，避免图片完全拖出视口。

## Canvas 清晰度与生命周期

- CSS 控制显示尺寸；实际 `canvas.width/height = CSS尺寸 × devicePixelRatio`。
- 每次绘制前设置 DPR transform 并清屏；容器尺寸变化时通过 `ResizeObserver` 重绘。
- 切换任务/输入项时取消或忽略旧图片加载结果，避免慢请求覆盖新选择。
- 图像跨域且需要像素读取/导出时配置正确 CORS；只绘制不读取像素的限制不同，也要实测目标浏览器。

## 区域绘制

多边形至少三个有效点才闭合绘制。逐点验证数值是否有限，必要时裁剪到 `[0,1]`。填充透明度、描边和颜色应可调，提供显隐开关；错误项仍展示原图和错误信息，便于复核。

若区域必须只叠加到某个输入角色，在结果契约中保存 `targetRole`，不要依赖 renderer 猜测。变化检测若在 A/B 两侧同时显示同一归一化多边形，应确保两图已配准。

## 可访问性与性能

画布交互之外提供键盘可达的上一项/下一项、显隐和复位按钮。对大量区域先做视口裁剪或离屏缓存；不要在每个 `pointermove` 中重新解析结果 JSON。连续重绘可合并到 `requestAnimationFrame`。
