import * as d3 from 'd3';

export function initCrispNamStructure(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Clear previous
    container.innerHTML = '';

    const width = 1000;
    const height = 450;
    const margin = { top: 40, right: 20, bottom: 40, left: 20 };
    const figureTitle = 'Figure 4: Schematic of CRISP-NAM Architecture';

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', 'auto')
        .attr('viewBox', `0 0 ${width} ${height}`)
        .style('max-width', '100%')
        .style('height', 'auto');
    
    // Accessibility title
    svg.append('title').text(figureTitle);

    // // Visible title
    svg.append('text')
        .attr('class', 'figure-title')
        .attr('x', width / 2)
        .attr('y', height - margin.bottom + 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '20px')
        .attr('font-weight', '600')
        .attr('fill', '#111')
        .text(figureTitle);


    // Definitions for markers
    const defs = svg.append('defs');
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#999');

    defs.append('marker')
        .attr('id', 'arrowhead-blue')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#4a90e2');

    defs.append('marker')
        .attr('id', 'arrowhead-red')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 8)
        .attr('refY', 0)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto')
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#e24a4a');

    // Layout parameters
    const xStart = 50;
    const yStart = 80;
    const xGap = 180;
    const yGap = 100;

    // Data structure for the diagram
    const features = [
        { id: 'x1', label: 'Covariate x₁', y: yStart },
        { id: 'x2', label: 'Covariate x₂', y: yStart + yGap },
        { id: 'xp', label: 'Covariate xₚ', y: yStart + yGap * 2.2 } // Gap for ellipsis
    ];

    // Draw layers

    // 1. Inputs
    const featureGroup = svg.append('g').attr('class', 'features');
    features.forEach(f => {
        drawBox(featureGroup, xStart, f.y, f.label, '#e3f2fd', '#2196f3', f.id);
    });
    // Ellipsis
    featureGroup.append('text').attr('x', xStart + 50).attr('y', yStart + yGap * 1.8).text('⋮').attr('class', 'arch-text');
    // Inputs
    featureGroup.append('text')
        .attr('x', xStart + 50)
        .attr('y', yStart - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text('Input');

    // 2. FeatureNets
    const netX = xStart + xGap;
    const netY = yStart;
    const netGroup = svg.append('g').attr('class', 'feature-nets');
    drawContainer(netGroup, netX - 15, netY - 15, 'FeatureNets', 'none', '#4caf50', 'feature-box', 100 + 30, yGap * 2.6 + 30);
    features.forEach(f => {
        drawBox(netGroup, netX, f.y, `f(${f.label.split(' ')[1]})`, '#e8f5e9', '#4caf50', 'net-' + f.id);
    });
    netGroup.append('text').attr('x', netX + 40).attr('y', yStart + yGap * 1.8).text('⋮').attr('class', 'arch-text');

    // 3. Projections (Split)
    const projX = netX + xGap;
    const projY = netY;
    const projGroup = svg.append('g').attr('class', 'projections');
    drawContainer(projGroup, projX - 15, projY - 23 - 15, 'Risk Projections', 'none', '#2196f3', 'proj-box', 100 + 30, yGap * 3 + 33);
    features.forEach((f, i) => {
        // Risk 1 (Blue, Up)
        drawBox(projGroup, projX, f.y - 23, `proj${i === 2 ? 'p' : i + 1},1`, 'white', '#2196f3', `proj1-${f.id}`);
        // Risk 2 (Red, Down)
        drawBox(projGroup, projX, f.y + 22, `proj${i === 2 ? 'p' : i + 1},2`, 'white', '#f44336', `proj2-${f.id}`);
    });

    projGroup.append('text').attr('x', projX + 32).attr('y', yStart + yGap * 1.85).text('⋮').attr('class', 'arch-text');


    // 4. Sums
    const sumX = projX + xGap + 80;
    const sumGroup = svg.append('g').attr('class', 'sums');
    const sum1Y = yStart + yGap * 0.5;
    const sum2Y = yStart + yGap * 2.0;
    drawContainer(sumGroup, sumX - 37, sum1Y - 30, '', 'none', '#f44336', 'sum-box', 70, yGap * 2 + 10);
    sumGroup.append('text')
        .attr('x', sumX)
        .attr('y', sum1Y - 60)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text('Additive Risk');
        sumGroup.append('text')
        .attr('x', sumX)
        .attr('y', sum1Y - 40)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text('Aggregation');
    drawCircle(sumGroup, sumX, sum1Y, '+', '#f5f5f5', '#2196f3', 'sum1');
    drawCircle(sumGroup, sumX, sum2Y, '+', '#f5f5f5', '#f44336', 'sum2');

    // 5. Final Risks
    const riskX =  sumX + xGap - 50;
    const riskGroup = svg.append('g').attr('class', 'risks');
    drawContainer(riskGroup, riskX - 15, sum1Y, '', 'none', '#2196f3', 'risk-box', 110, yGap * 1.5);
    riskGroup.append('text')
        .attr('x', riskX + 45)
        .attr('y', sum1Y - 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text('Cause Specific');
    riskGroup.append('text')
        .attr('x', riskX + 45)
        .attr('y', sum1Y - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '18px')
        .attr('font-weight', '600')
        .attr('fill', '#333')
        .text('Log hazard Calculation');
    drawBox(riskGroup, riskX, sum1Y + 20, 'η₁(x)', '#e3f2fd', '#2196f3', 'eta1', 80);
    drawBox(riskGroup, riskX, sum2Y - 60, 'η₂(x)', '#ffebee', '#f44336', 'eta2', 80);


    // --- Connections ---
    const edgesLayer = svg.append('g').attr('class', 'edges').lower();

    // x -> Net -> h
    features.forEach(f => {
        drawArrow(edgesLayer, xStart + 100, f.y + 20, netX, f.y + 20, '#999'); // x -> net
        drawArrow(edgesLayer, netX, f.y + 20, projX, f.y + 20, '#999');   // net -> proj

        // proj -> sum1
        const g1Box = { x: projX + 80, y: f.y - 5 };
        drawArrowCurve(edgesLayer, g1Box.x, g1Box.y, sumX - 22, sum1Y, '#4a90e2');

        // proj -> sum2
        const g2Box = { x: projX + 80, y: f.y + 45 };
        drawArrowCurve(edgesLayer, g2Box.x, g2Box.y, sumX - 22, sum2Y, '#e24a4a');
    });

    // sum -> eta
    drawArrow(edgesLayer, sumX + 20, sum1Y, riskX, sum1Y + 43, '#4a90e2', 'arrowhead-blue');

    drawArrow(edgesLayer, sumX + 20, sum2Y, riskX, sum2Y - 43, '#e24a4a', 'arrowhead-red');

    // Helpers
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
            .attr('y', y - 10)
            .attr('text-anchor', 'middle')
            .attr('font-size', '18px')
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
    }

    function drawArrow(group, x1, y1, x2, y2, color, markerId = 'arrowhead') {
        const line = group.append('line')
            .attr('x1', x1)
            .attr('y1', y1)
            .attr('x2', x2)
            .attr('y2', y2)
            .attr('stroke', color)
            .attr('stroke-width', 2)
            .attr('marker-end', `url(#${markerId})`)
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

    function drawArrowCurve(group, x1, y1, x2, y2, color) {
        const midX = (x1 + x2) / 2;
        const path = d3.path();
        path.moveTo(x1, y1);
        path.bezierCurveTo(midX, y1, midX, y2, x2, y2);

        const curve = group.append('path')
            .attr('d', path.toString())
            .attr('stroke', color)
            .attr('stroke-width', 2)
            .attr('fill', 'none')
            .attr('marker-end', color.includes('e24a4a') ? 'url(#arrowhead-red)' : 'url(#arrowhead-blue)')
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

    // Animation: Flow particles
    function animateParticles() {
        features.forEach((f, i) => {
            const particle = svg.append('circle')
                .attr('r', 4)
                .attr('fill', '#333')
                .attr('cx', xStart + 100)
                .attr('cy', f.y + 20);

            particle.transition()
                .duration(1000)
                .attr('cx', netX)
                .transition()
                .duration(1000)
                .attr('cx', projX)
                .transition()
                .on('end', () => {
                    // Split
                    const p1 = svg.append('circle').attr('r', 3).attr('fill', '#2196f3').attr('cx', projX + 60).attr('cy', f.y + 20);
                    const p2 = svg.append('circle').attr('r', 3).attr('fill', '#f44336').attr('cx', projX + 60).attr('cy', f.y + 20);

                    p1.transition().duration(1000)
                        .attrTween('transform', function () {
                            const startX = projX + 60; const startY = f.y + 20;
                            const endX = sumX; const endY = sum1Y;
                            return (t) => {
                                const x = startX + (endX - startX) * t;
                                // Simple easing for y
                                const y = startY + (endY - startY) * t;
                                return `translate(${x - startX}, ${y - startY})`;
                            };
                        })
                        .remove();

                    p2.transition().duration(1000)
                        .attrTween('transform', function () {
                            const startX = projX + 60; const startY = f.y + 20;
                            const endX = sumX; const endY = sum2Y;
                            return (t) => {
                                const x = startX + (endX - startX) * t;
                                const y = startY + (endY - startY) * t;
                                return `translate(${x - startX}, ${y - startY})`;
                            };
                        })
                        .remove();

                    particle.remove();
                });
        });

        setTimeout(animateParticles, 4000);
    }
}
