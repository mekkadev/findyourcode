export function drawSeries(canvas, points, options = {}) {
  const ctx = canvas.getContext("2d");
  const scaleX = canvas.width / (points.length - 1 || 1);
  const maxY = Math.max(...points.map((p) => p.y), 1);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  points.forEach((point, i) => {
    const y = canvas.height - (point.y / maxY) * canvas.height;
    i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * scaleX, y);
  });
  ctx.strokeStyle = options.color ?? "#3b82f6";
  ctx.stroke();
}

export const formatTick = (value) =>
  value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value);
