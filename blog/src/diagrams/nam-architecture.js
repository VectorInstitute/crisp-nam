import * as d3 from 'd3';

export function initNamStructure(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Clear previous
    container.innerHTML = '';

    const width = 800;
    const height = 400; // Compact height
    const margin = { top: 40, right: 40, bottom: 80, left: 40 };
    // --- Figure title and description (bottom center) ---
    const figureTitle = 'Figure 3: Schematic of Neural Additive Model (NAM) Architecture'
    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', 'auto')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .style('max-width', '100%')
        .style('height', 'auto');

        // Accessibility title
    svg.append('title').text(figureTitle);

    // Definitions for markers
    const defs = svg.append('defs');
    defs.append('marker')
        .attr('id', 'arrowhead-nam')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#666');
    
    // // Visible title
    svg.append('text')
        .attr('class', 'figure-title')
        .attr('x', width / 2)
        .attr('y', height - margin.bottom + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '16px')
        .attr('font-weight', '600')
        .attr('fill', '#111')
        .text(figureTitle);

    // Layout
    const xStart = 50;
    const yStart = 60;
    const yGap = 80;

    // Nodes Data
    const features = [
        { id: 'x1', label: 'Covariate 1', y: yStart },
        { id: 'x2', label: 'Covariate 2', y: yStart + yGap },
        { id: 'x3', label: 'Covariate 3', y: yStart + yGap * 2 }
    ];

    const finalFeature = { id: 'xN', label: 'Covariate N', y: yStart + yGap * 2.8 };


    // Layers X Coordinates
    const xFeatures = xStart;
    const xNets = xStart + 180;
    const xSum = xNets + 280;
    const xPred = xSum + 120;

    // Draw Groups
    const edgesGroup = svg.append('g').attr('class', 'edges');
    const nodesGroup = svg.append('g').attr('class', 'nodes');

    // --- Draw Nodes ---

    // 1. Features
    features.forEach(f => {
        drawBox(nodesGroup, xFeatures, f.y, f.label, '#e3f2fd', '#2196f3', f.id);
    });
    // Ellipsis for features
    nodesGroup.append('text').attr('x', xFeatures + 50).attr('y', yStart + yGap * 2.7).text('⋮').attr('class', 'arch-text').attr('text-anchor', 'middle');
    nodesGroup.append('text')
        .attr('x', xStart + 50)
        .attr('y', yStart - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '14px')
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text('Input');
    //Draw final feature
    drawBox(nodesGroup, xFeatures, finalFeature.y, finalFeature.label, '#e3f2fd', '#2196f3', finalFeature.id);

    // 2. Neural Nets
    features.forEach(f => {
        drawBox(nodesGroup, xNets, f.y, `Neural Net ${f.id.replace('x', '')}`, '#e8f5e9', '#4caf50', `net-${f.id}`, 120);
    });
    nodesGroup.append('text').attr('x', xNets + 60).attr('y', yStart + yGap * 2.7).text('⋮').attr('class', 'arch-text').attr('text-anchor', 'middle');

    // Draw final neural net
    drawBox(nodesGroup, xNets, finalFeature.y, `Neural Net ${finalFeature.id.replace('x', '')}`, '#e8f5e9', '#4caf50', `net-${finalFeature.id}`, 120);

    // 3. Sum Node
    const ySum = yStart + yGap * 1.5; // Centered
    const sumBoxSize = 50;
    // Transparent box centered at (xSum, ySum)
    drawContainer(nodesGroup, xSum - sumBoxSize / 2, ySum - sumBoxSize / 2, 'Aggregate', 'none', '#9e9e9e', 'sum-box', sumBoxSize, sumBoxSize);
    drawCircle(nodesGroup, xSum, ySum, '+', '#f5f5f5', '#9e9e9e', 'sum');

    // 4. Prediction Node
    const yPred = ySum;
    const predBoxSize = 50;
    const predRadius = 20;

    drawContainer(nodesGroup, xPred - predBoxSize / 2, yPred - predBoxSize / 2, 'Prediction', 'none', '#e91e63', 'pred-box', predBoxSize, predBoxSize);
    drawCircle(nodesGroup, xPred, yPred, 'σ', '#fce4ec', '#e91e63', 'pred', predRadius);

    // --- Draw Edges ---

    features.forEach(f => {
        // Feature -> Net
        drawArrow(edgesGroup, xFeatures + 100, f.y + 20, xNets, f.y + 20);

        // Net -> Function
        drawArrowCurve(edgesGroup, xNets + 120, f.y + 20, xSum-27, ySum);

    });

    // Final Feature -> Final Net Arrows
    drawArrow(edgesGroup, xFeatures + 100, finalFeature.y + 20, xNets, finalFeature.y + 20);
    drawArrowCurve(edgesGroup, xNets + 120, finalFeature.y + 20, xSum-27, ySum);


    // Sum -> Prediction
    drawArrow(edgesGroup, xSum + 27, ySum, xPred - 27, yPred);


    // --- Helpers ---

    function drawBox(group, x, y, text, fill, stroke, id, w = 100, h = 40) {
        const g = group.append('g').attr('id', id);
        g.append('rect')
            .attr('x', x)
            .attr('y', y)
            .attr('width', w)
            .attr('height', h)
            .attr('rx', 5)
            .attr('ry', 5)
            .attr('fill', fill)
            .attr('stroke', stroke)
            .attr('stroke-width', 2);

        g.append('text')
            .attr('x', x + w / 2)
            .attr('y', y + h / 2 + 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '14px')
            .attr('fill', '#333')
            .text(text);
        return g;
    }

    function drawContainer(group, x, y, text, fill, stroke, id, w = 100, h = 40) {
        const g = group.append('g').attr('id', id);
        g.append('rect')
            .attr('x', x)
            .attr('y', y)
            .attr('width', w)
            .attr('height', h)
            .attr('rx', 5)
            .attr('ry', 5)
            .attr('fill', fill)
            .attr('stroke', stroke)
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '6 4');

        g.append('text')
            .attr('x', x + w / 2)
            .attr('y', y - 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '14px')
            .attr('font-weight', '600')
            .attr('fill', '#333')
            .text(text);
        return g;
    }

    function drawCircle(group, x, y, text, fill, stroke, id, r = 20) {
        const g = group.append('g').attr('id', id);
        g.append('circle')
            .attr('cx', x)
            .attr('cy', y)
            .attr('r', r)
            .attr('fill', fill)
            .attr('stroke', stroke)
            .attr('stroke-width', 2);

        g.append('text')
            .attr('x', x)
            .attr('y', y + 5)
            .attr('text-anchor', 'middle')
            .attr('font-size', '20px')
            .text(text);
        return g;
    }

    function drawArrow(group, x1, y1, x2, y2) {
        const line = group.append('line')
            .attr('x1', x1)
            .attr('y1', y1)
            .attr('x2', x2)
            .attr('y2', y2)
            .attr('stroke', '#666')
            .attr('stroke-width', 2)
            .attr('marker-end', 'url(#arrowhead-nam)')
            .attr('stroke-linecap', 'round')
            .attr('stroke-dasharray', '8 12')   // dash pattern
            .attr('stroke-dashoffset', 0);

         // Continuous motion from start to end
         const speed = 80; // px per second; increase for quicker motion
         d3.timer((elapsed) => {
             const offset = -(elapsed / 1000) * speed;
             line.attr('stroke-dashoffset', offset);
         });
    }

    function drawArrowCurve(group, x1, y1, x2, y2) {
        const midX = (x1 + x2) / 2;
        const path = d3.path();
        path.moveTo(x1, y1);
        path.bezierCurveTo(midX, y1, midX, y2, x2, y2);

        const curve = group.append('path')
            .attr('d', path.toString())
            .attr('stroke', '#666')
            .attr('stroke-width', 2)
            .attr('fill', 'none')
            .attr('marker-end', 'url(#arrowhead-nam)')
            .attr('stroke-linecap', 'round')
            .attr('stroke-dasharray', '8 12')
            .attr('stroke-dashoffset', 0);


        // Continuous motion along the curve
        const speed = 80; // px per second
        d3.timer((elapsed) => {
            const offset = -(elapsed / 1000) * speed;
            curve.attr('stroke-dashoffset', offset);
        });
    }

    // Optional simple animation on load
    d3.selectAll('.nodes rect').style('opacity', 0).transition().duration(800).style('opacity', 1);
}
