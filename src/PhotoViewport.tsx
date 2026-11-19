import { useEffect, useRef, useState, type ReactNode } from "react";

type View = { scale: number; x: number; y: number };
const fit: View = { scale: 1, x: 0, y: 0 };

export function PhotoViewport({ children }: { children: ReactNode }) {
  const stage = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const view = useRef<View>(fit);
  const drag = useRef<{ id: number; x: number; y: number; view: View } | undefined>(undefined);
  const [display, setDisplay] = useState<View>(fit);

  function update(next: View) {
    const area = stage.current, photo = content.current;
    if (!area || !photo) return;
    const limitX = Math.max(0, (photo.offsetWidth * next.scale - area.clientWidth) / 2);
    const limitY = Math.max(0, (photo.offsetHeight * next.scale - area.clientHeight) / 2);
    view.current = {
      scale: next.scale,
      x: Math.max(-limitX, Math.min(limitX, next.x)),
      y: Math.max(-limitY, Math.min(limitY, next.y)),
    };
    setDisplay(view.current);
  }

  useEffect(() => {
    const area = stage.current!, photo = content.current!;
    function wheel(event: WheelEvent) {
      event.preventDefault();
      const old = view.current;
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? area.clientHeight : 1);
      const scale = Math.max(1, Math.min(8, old.scale * Math.exp(-Math.max(-100, Math.min(100, delta)) * .002)));
      const box = area.getBoundingClientRect();
      const x = event.clientX - box.left - box.width / 2;
      const y = event.clientY - box.top - box.height / 2;
      update({ scale, x: x - (x - old.x) * scale / old.scale, y: y - (y - old.y) * scale / old.scale });
    }
    // A native non-passive listener keeps the wheel inside the photo viewer.
    area.addEventListener("wheel", wheel, { passive: false });
    const resize = new ResizeObserver(() => update(view.current));
    resize.observe(area); resize.observe(photo);
    return () => { area.removeEventListener("wheel", wheel); resize.disconnect(); };
  }, []);

  return <div className="photo-viewport">
    <div ref={stage} className={`stage${display.scale > 1 ? " zoomed" : ""}`}
      onDragStart={event => event.preventDefault()}
      onPointerDown={event => {
        if (event.button !== 0 || view.current.scale === 1) return;
        drag.current = { id: event.pointerId, x: event.clientX, y: event.clientY, view: view.current };
      }}
      onPointerMove={event => {
        const start = drag.current;
        if (start?.id !== event.pointerId || Math.hypot(event.clientX - start.x, event.clientY - start.y) < 3) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        update({ ...start.view, x: start.view.x + event.clientX - start.x, y: start.view.y + event.clientY - start.y });
      }}
      onPointerUp={event => {
        if (drag.current?.id === event.pointerId) {
          drag.current = undefined;
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        }
      }}
      onPointerLeave={event => { if (!event.currentTarget.hasPointerCapture(event.pointerId)) drag.current = undefined; }}
      onPointerCancel={() => { drag.current = undefined; }}
      onLostPointerCapture={() => { drag.current = undefined; }}>
      <div ref={content} className="zoom-content" style={{ transform: `translate(${display.x}px, ${display.y}px) scale(${display.scale})` }}>{children}</div>
    </div>
    {display.scale > 1 && <button className="fit-photo" onClick={() => update(fit)} aria-label="Fit photo to view">Fit</button>}
  </div>;
}
