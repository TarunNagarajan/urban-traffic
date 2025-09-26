import React, { useRef, useEffect, useMemo } from 'react'

type Intersection = {
  id: string
  phase: number
  totalQueue: number
}

type IntersectionVisualizationProps = {
  intersections: Intersection[]
  width?: number
  height?: number
}

export function IntersectionVisualization({ 
  intersections, 
  width = 400, 
  height = 300 
}: IntersectionVisualizationProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const drawIntersection = useMemo(() => {
    return (ctx: CanvasRenderingContext2D, intersection: Intersection, x: number, y: number, size: number) => {
      const centerX = x + size / 2
      const centerY = y + size / 2
      const roadWidth = size * 0.3
      const laneWidth = roadWidth / 2

      // Clear the area
      ctx.clearRect(x, y, size, size)

      ctx.fillStyle = '#333'
      ctx.fillRect(x, centerY - roadWidth/2, size, roadWidth) // Horizontal road
      ctx.fillRect(centerX - roadWidth/2, y, roadWidth, size) // Vertical road

      // Draw lanes
      ctx.fillStyle = '#666'
      // Horizontal lanes
      ctx.fillRect(x, centerY - roadWidth/2, size, laneWidth) // Top lane
      ctx.fillRect(x, centerY, size, laneWidth) // Bottom lane
      // Vertical lanes  
      ctx.fillRect(centerX - roadWidth/2, y, laneWidth, size) // Left lane
      ctx.fillRect(centerX, y, laneWidth, size) // Right lane

      // Draw lane markings
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.setLineDash([10, 5])
      ctx.beginPath()
      ctx.moveTo(x, centerY)
      ctx.lineTo(x + size, centerY)
      ctx.moveTo(centerX, y)
      ctx.lineTo(centerX, y + size)
      ctx.stroke()
      ctx.setLineDash([])

      // Draw traffic lights
      const lightSize = size * 0.08
      const lightSpacing = size * 0.15

      // North light
      const northX = centerX - lightSize/2
      const northY = y + lightSpacing
      drawTrafficLight(ctx, northX, northY, lightSize, intersection.phase === 0 ? 'green' : 'red')

      // South light
      const southX = centerX - lightSize/2
      const southY = y + size - lightSpacing - lightSize
      drawTrafficLight(ctx, southX, southY, lightSize, intersection.phase === 0 ? 'red' : 'green')

      // East light
      const eastX = x + size - lightSpacing - lightSize
      const eastY = centerY - lightSize/2
      drawTrafficLight(ctx, eastX, eastY, lightSize, intersection.phase === 0 ? 'red' : 'green')

      // West light
      const westX = x + lightSpacing
      const westY = centerY - lightSize/2
      drawTrafficLight(ctx, westX, westY, lightSize, intersection.phase === 0 ? 'green' : 'red')

      // Draw vehicles (represented as small rectangles)
      drawVehicles(ctx, intersection, x, y, size, roadWidth, centerX, centerY)

      // Draw intersection ID
      ctx.fillStyle = '#000'
      ctx.font = '12px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(intersection.id, centerX, centerY + 4)

      // Draw queue length
      ctx.fillStyle = '#666'
      ctx.font = '10px Arial'
      ctx.fillText(`Q: ${intersection.totalQueue}`, centerX, y + size + 15)
    }
  }, [])

  const drawTrafficLight = (ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: 'red' | 'green' | 'yellow') => {
    // Light housing
    ctx.fillStyle = '#222'
    ctx.fillRect(x, y, size, size * 1.5)
    
    // Light
    ctx.fillStyle = color === 'red' ? '#ff4444' : color === 'green' ? '#44ff44' : '#ffff44'
    ctx.beginPath()
    ctx.arc(x + size/2, y + size/2, size/3, 0, 2 * Math.PI)
    ctx.fill()
    
    // Light glow
    ctx.shadowColor = color === 'red' ? '#ff4444' : color === 'green' ? '#44ff44' : '#ffff44'
    ctx.shadowBlur = 5
    ctx.fill()
    ctx.shadowBlur = 0
  }

  const drawVehicles = (ctx: CanvasRenderingContext2D, intersection: Intersection, x: number, y: number, size: number, roadWidth: number, centerX: number, centerY: number) => {
    const vehicleCount = Math.min(intersection.totalQueue, 8) // Limit visual vehicles
    const vehicleSize = size * 0.04
    const laneWidth = roadWidth / 2

    ctx.fillStyle = '#4a90e2'
    
    for (let i = 0; i < vehicleCount; i++) {
      const position = (i + 1) / (vehicleCount + 1)
      
      // Place vehicles on different approaches based on phase
      if (intersection.phase === 0) {
        // North-South green
        if (i % 2 === 0) {
          // North approach
          const vehicleX = centerX - laneWidth/2 + (i % 2) * laneWidth
          const vehicleY = y + (1 - position) * (centerY - y - roadWidth/2)
          ctx.fillRect(vehicleX, vehicleY, vehicleSize, vehicleSize * 1.5)
        } else {
          // South approach
          const vehicleX = centerX - laneWidth/2 + (i % 2) * laneWidth
          const vehicleY = centerY + roadWidth/2 + position * (y + size - centerY - roadWidth/2)
          ctx.fillRect(vehicleX, vehicleY, vehicleSize, vehicleSize * 1.5)
        }
      } else {
        // East-West green
        if (i % 2 === 0) {
          // East approach
          const vehicleX = centerX + roadWidth/2 + position * (x + size - centerX - roadWidth/2)
          const vehicleY = centerY - laneWidth/2 + (i % 2) * laneWidth
          ctx.fillRect(vehicleX, vehicleY, vehicleSize * 1.5, vehicleSize)
        } else {
          // West approach
          const vehicleX = x + (1 - position) * (centerX - x - roadWidth/2)
          const vehicleY = centerY - laneWidth/2 + (i % 2) * laneWidth
          ctx.fillRect(vehicleX, vehicleY, vehicleSize * 1.5, vehicleSize)
        }
      }
    }
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = width
    canvas.height = height

    // Clear canvas
    ctx.clearRect(0, 0, width, height)

    // Draw background
    ctx.fillStyle = '#f0f0f0'
    ctx.fillRect(0, 0, width, height)

    // Calculate grid layout
    const cols = Math.ceil(Math.sqrt(intersections.length))
    const rows = Math.ceil(intersections.length / cols)
    const cellWidth = width / cols
    const cellHeight = height / rows
    const intersectionSize = Math.min(cellWidth, cellHeight) * 0.8

    // Draw each intersection
    intersections.forEach((intersection, index) => {
      const col = index % cols
      const row = Math.floor(index / cols)
      const x = col * cellWidth + (cellWidth - intersectionSize) / 2
      const y = row * cellHeight + (cellHeight - intersectionSize) / 2
      
      drawIntersection(ctx, intersection, x, y, intersectionSize)
    })
  }, [intersections, width, height, drawIntersection])

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, overflow: 'hidden' }}>
      <div style={{ padding: '8px 12px', background: '#f8f9fa', borderBottom: '1px solid #ddd' }}>
        <h4 style={{ margin: 0, fontSize: '14px', fontWeight: 600 }}>Traffic Intersections</h4>
      </div>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ display: 'block', maxWidth: '100%', height: 'auto' }}
      />
    </div>
  )
}
