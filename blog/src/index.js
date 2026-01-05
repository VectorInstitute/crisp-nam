/**
 * Main entry point for the blog
 */

import { initVisualizations } from './diagrams/index';
import { initModelInference } from './model-inference';

// Initialize all the interactive visualizations when the page loads
document.addEventListener('DOMContentLoaded', () => {
  initVisualizations();
  initModelInference();
});