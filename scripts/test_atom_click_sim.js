/**
 * test_atom_click_sim.js
 *
 * Self-contained Node.js simulation test for 3D viewer atom click & hover interactions:
 * 1. Clicking in Auto mode switches manualMode.checked = true and autoMode.checked = false.
 * 2. Sets atomDropdown.value to clicked atom's label and triggers change event.
 * 3. Retains any pre-selected substitute-element (Action) and enables execute-btn.
 * 4. Guards against clicking when an operation is processing (button-processing class or currentAbortController !== null).
 * 5. Guards against clicking when on the "Insert New" tab.
 * 6. Verifies hover sets cursor: pointer in both Auto and Manual modes when in Modify tab and not processing.
 */

const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');

// --- Minimal Mock DOM Implementation ---

class MockClassList {
    constructor() {
        this.classes = new Set();
    }
    add(c) { this.classes.add(c); }
    remove(c) { this.classes.delete(c); }
    contains(c) { return this.classes.has(c); }
    toString() { return Array.from(this.classes).join(' '); }
}

class MockElement {
    constructor(tagName, id = '') {
        this.tagName = tagName.toUpperCase();
        this.id = id;
        this.classList = new MockClassList();
        this.style = {};
        this.attributes = {};
        this.children = [];
        this.value = '';
        this.checked = false;
        this.disabled = false;
        this.textContent = '';
        this.listeners = {};
    }

    appendChild(child) {
        this.children.push(child);
        return child;
    }

    querySelector(selector) {
        // Handle option[value="XYZ"]
        const optionMatch = selector.match(/option\[value=["']?([^"']+)["']?\]/i);
        if (optionMatch) {
            const val = optionMatch[1];
            return this.children.find(c => c.tagName === 'OPTION' && c.value === val) || null;
        }
        return null;
    }

    querySelectorAll(selector) {
        if (selector === 'option') {
            return this.children.filter(c => c.tagName === 'OPTION');
        }
        return [];
    }

    addEventListener(event, handler) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(handler);
    }

    dispatchEvent(event) {
        const handlers = this.listeners[event.type] || [];
        for (const h of handlers) {
            h.call(this, event);
        }
        return true;
    }
}

class MockEvent {
    constructor(type) {
        this.type = type;
    }
}

class MockDocument {
    constructor() {
        this.elements = new Map();
    }

    registerElement(el) {
        if (el.id) {
            this.elements.set(el.id, el);
        }
        return el;
    }

    getElementById(id) {
        return this.elements.get(id) || null;
    }

    createElement(tagName) {
        return new MockElement(tagName);
    }
}

// Helper to extract function code from HTML/JS
function extractFunction(source, fnName) {
    const idx = source.indexOf(`function ${fnName}(`);
    if (idx === -1) throw new Error(`Function ${fnName} not found in source`);
    const openBrace = source.indexOf('{', idx);
    let depth = 1;
    let i = openBrace + 1;
    while (i < source.length && depth > 0) {
        if (source[i] === '{') depth++;
        else if (source[i] === '}') depth--;
        i++;
    }
    return source.slice(idx, i);
}

// Read templates/index.html
const indexPath = path.join(__dirname, '..', 'templates', 'index.html');
const indexSource = fs.readFileSync(indexPath, 'utf8');

const setupViewerAtomInteractionsCode = extractFunction(indexSource, 'setupViewerAtomInteractions');
const setupOperationListenersCode = extractFunction(indexSource, 'setupOperationListeners');
const updateExecuteButtonCode = extractFunction(indexSource, 'updateExecuteButton');

function setupTestEnvironment() {
    const doc = new MockDocument();

    // Create required DOM elements
    const tabModify = doc.registerElement(new MockElement('button', 'tab-modify'));
    tabModify.classList.add('active');

    const tabInsert = doc.registerElement(new MockElement('button', 'tab-insert'));
    const modifyWorkflow = doc.registerElement(new MockElement('div', 'modify-workflow'));
    const insertWorkflow = doc.registerElement(new MockElement('div', 'insert-workflow'));

    const manualMode = doc.registerElement(new MockElement('input', 'manual-mode'));
    manualMode.checked = false;

    const autoMode = doc.registerElement(new MockElement('input', 'auto-mode'));
    autoMode.checked = true;

    const manualSelector = doc.registerElement(new MockElement('div', 'manual-selector'));
    manualSelector.style.display = 'none';

    const autoSelector = doc.registerElement(new MockElement('div', 'auto-selector'));
    autoSelector.style.display = 'block';

    const atomDropdown = doc.registerElement(new MockElement('select', 'element-dropdown'));
    const emptyOpt = new MockElement('option');
    emptyOpt.value = '';
    emptyOpt.textContent = 'Choose an atom...';
    atomDropdown.appendChild(emptyOpt);

    const labels = ['Cu0', 'Cu1', 'O0', 'O1'];
    labels.forEach(l => {
        const opt = new MockElement('option');
        opt.value = l;
        opt.textContent = l;
        atomDropdown.appendChild(opt);
    });

    const elementTypeDropdown = doc.registerElement(new MockElement('select', 'element-type-dropdown'));
    const emptyTypeOpt = new MockElement('option');
    emptyTypeOpt.value = '';
    elementTypeDropdown.appendChild(emptyTypeOpt);
    ['Cu', 'O'].forEach(el => {
        const opt = new MockElement('option');
        opt.value = el;
        opt.textContent = el;
        elementTypeDropdown.appendChild(opt);
    });

    const substituteElement = doc.registerElement(new MockElement('select', 'substitute-element'));
    const emptySubOpt = new MockElement('option');
    emptySubOpt.value = '';
    substituteElement.appendChild(emptySubOpt);
    ['Au', 'DELETE'].forEach(sub => {
        const opt = new MockElement('option');
        opt.value = sub;
        opt.textContent = sub;
        substituteElement.appendChild(opt);
    });

    const insertAutoMode = doc.registerElement(new MockElement('input', 'insert-auto-mode'));
    const insertManualMode = doc.registerElement(new MockElement('input', 'insert-manual-mode'));
    const insertElementDropdown = doc.registerElement(new MockElement('select', 'insert-element'));
    const insertManualSelector = doc.registerElement(new MockElement('div', 'insert-manual-selector'));
    const insertCandidateDropdown = doc.registerElement(new MockElement('select', 'insert-candidate'));
    const executeBtn = doc.registerElement(new MockElement('button', 'execute-btn'));
    executeBtn.disabled = true;
    executeBtn.textContent = 'Execute';
    const resetOperationsBtn = doc.registerElement(new MockElement('button', 'reset-operations-btn'));

    let viewerClickCallback = null;
    let viewerHoverInCallback = null;
    let viewerHoverOutCallback = null;
    let highlightedAtom = null;

    const mockViewer3D = {
        setClickable: (sel, flag, cb) => {
            viewerClickCallback = cb;
        },
        setHoverable: (sel, flag, inCb, outCb) => {
            viewerHoverInCallback = inCb;
            viewerHoverOutCallback = outCb;
        },
        addLabel: () => ({}),
        render: () => {}
    };

    const container = {
        style: { cursor: 'default' }
    };

    const context = {
        document: doc,
        Event: MockEvent,
        window: {
            viewer3D: mockViewer3D,
            currentLabels: labels,
            currentOperationMode: 'modify',
            originalSupercellData: null,
            originalLabels: []
        },
        currentAbortController: null,
        highlightManualAtom: (label) => {
            highlightedAtom = label;
        },
        clearManualAtomHighlight: () => {
            highlightedAtom = null;
        },
        clearGhostAtom: () => {},
        console: console
    };

    vm.createContext(context);

    // Run the extracted functions
    vm.runInContext(updateExecuteButtonCode, context);
    vm.runInContext(setupOperationListenersCode, context);
    vm.runInContext(setupViewerAtomInteractionsCode, context);

    // Expose updateExecuteButton globally on context
    context.updateExecuteButton = context.updateExecuteButton || context.window.updateExecuteButton;

    // Initialize listeners
    vm.runInContext('setupOperationListeners();', context);
    vm.runInContext('setupViewerAtomInteractions();', context);

    return {
        context,
        doc,
        elements: {
            tabModify,
            tabInsert,
            manualMode,
            autoMode,
            manualSelector,
            autoSelector,
            atomDropdown,
            elementTypeDropdown,
            substituteElement,
            executeBtn
        },
        container,
        getViewerClickCallback: () => viewerClickCallback,
        getViewerHoverInCallback: () => viewerHoverInCallback,
        getViewerHoverOutCallback: () => viewerHoverOutCallback,
        getHighlightedAtom: () => highlightedAtom
    };
}

// --- Run All 6 Verification Checks ---

console.log('🧪 Starting 3D Atom Click & Hover Simulation Tests...\n');

// Test 1: Clicking in Auto mode switches manualMode.checked = true and autoMode.checked = false
{
    console.log('Test 1: Clicking in Auto mode switches to Manual mode');
    const env = setupTestEnvironment();
    const { elements, getViewerClickCallback } = env;

    assert.strictEqual(elements.manualMode.checked, false, 'Initial manualMode should be false');
    assert.strictEqual(elements.autoMode.checked, true, 'Initial autoMode should be true');

    const clickCb = getViewerClickCallback();
    assert(typeof clickCb === 'function', 'setClickable callback must be registered');

    // Simulate clicking atom index 1 ('Cu1')
    clickCb({ index: 1 }, env.context.window.viewer3D, {}, env.container);

    assert.strictEqual(elements.manualMode.checked, true, 'manualMode should now be checked (true)');
    assert.strictEqual(elements.autoMode.checked, false, 'autoMode should now be unchecked (false)');
    console.log('  ✅ Test 1 Passed: Successfully switched from Auto to Manual mode.');
}

// Test 2: Sets atomDropdown.value to the clicked atom's label and triggers change event
{
    console.log('\nTest 2: Sets atomDropdown.value to clicked atom label and triggers change');
    const env = setupTestEnvironment();
    const { elements, getViewerClickCallback, getHighlightedAtom } = env;

    let changeEventFired = false;
    elements.atomDropdown.addEventListener('change', () => {
        changeEventFired = true;
    });

    const clickCb = getViewerClickCallback();
    clickCb({ index: 1 }, env.context.window.viewer3D, {}, env.container);

    assert.strictEqual(elements.atomDropdown.value, 'Cu1', 'atomDropdown value should be set to "Cu1"');
    assert.strictEqual(changeEventFired, true, 'change event on atomDropdown must have fired');
    assert.strictEqual(getHighlightedAtom(), 'Cu1', 'Manual atom highlight should be triggered for "Cu1"');
    console.log('  ✅ Test 2 Passed: atomDropdown updated and change event fired.');
}

// Test 3: Retains any pre-selected substitute-element (Action) and enables execute-btn
{
    console.log('\nTest 3: Retains pre-selected substitute-element and enables execute-btn');
    const env = setupTestEnvironment();
    const { elements, getViewerClickCallback } = env;

    // In Auto mode, user selects substitute action 'Au'
    elements.substituteElement.value = 'Au';
    elements.substituteElement.dispatchEvent(new MockEvent('change'));
    assert.strictEqual(elements.executeBtn.disabled, true, 'Execute button should be disabled before target atom is selected');

    // User clicks atom index 1 ('Cu1')
    const clickCb = getViewerClickCallback();
    clickCb({ index: 1 }, env.context.window.viewer3D, {}, env.container);

    assert.strictEqual(elements.substituteElement.value, 'Au', 'Pre-selected substitute element should be retained');
    assert.strictEqual(elements.executeBtn.disabled, false, 'Execute button should be enabled');
    assert(elements.executeBtn.textContent.includes('Cu1 → Au'), `Execute button text should show operation, got "${elements.executeBtn.textContent}"`);
    console.log('  ✅ Test 3 Passed: Action preserved and execute-btn enabled with correct text.');
}

// Test 4: Guards against clicking when an operation is processing
{
    console.log('\nTest 4: Guards against clicking during processing (button-processing or currentAbortController)');
    const env = setupTestEnvironment();
    const { elements, getViewerClickCallback, context } = env;
    const clickCb = getViewerClickCallback();

    // 4A: button-processing class
    elements.executeBtn.classList.add('button-processing');
    clickCb({ index: 1 }, env.context.window.viewer3D, {}, env.container);

    assert.strictEqual(elements.manualMode.checked, false, 'Must not switch mode while button-processing is active');
    assert.strictEqual(elements.atomDropdown.value, '', 'Must not set atomDropdown while button-processing is active');

    elements.executeBtn.classList.remove('button-processing');

    // 4B: currentAbortController is not null
    context.currentAbortController = { abort: () => {} };
    clickCb({ index: 1 }, env.context.window.viewer3D, {}, env.container);

    assert.strictEqual(elements.manualMode.checked, false, 'Must not switch mode while currentAbortController is active');
    assert.strictEqual(elements.atomDropdown.value, '', 'Must not set atomDropdown while currentAbortController is active');

    context.currentAbortController = null;
    console.log('  ✅ Test 4 Passed: Clicking guarded against processing states.');
}

// Test 5: Guards against clicking when on the "Insert New" tab
{
    console.log('\nTest 5: Guards against clicking when on the "Insert New" tab');
    const env = setupTestEnvironment();
    const { elements, getViewerClickCallback } = env;
    const clickCb = getViewerClickCallback();

    // Switch to Insert tab
    elements.tabModify.classList.remove('active');
    elements.tabInsert.classList.add('active');

    clickCb({ index: 1 }, env.context.window.viewer3D, {}, env.container);

    assert.strictEqual(elements.manualMode.checked, false, 'Must not switch mode when on Insert tab');
    assert.strictEqual(elements.atomDropdown.value, '', 'Must not select atom when on Insert tab');
    console.log('  ✅ Test 5 Passed: Clicking ignored when not on Modify tab.');
}

// Test 6: Verifies hover sets cursor: pointer in both Auto and Manual modes when in Modify tab and not processing
{
    console.log('\nTest 6: Hover cursor styling in Auto and Manual modes and guards');
    const env = setupTestEnvironment();
    const { elements, container, getViewerHoverInCallback, getViewerHoverOutCallback, context } = env;
    const hoverIn = getViewerHoverInCallback();
    const hoverOut = getViewerHoverOutCallback();

    assert(typeof hoverIn === 'function', 'HoverIn callback must be registered');
    assert(typeof hoverOut === 'function', 'HoverOut callback must be registered');

    // 6A: In Auto mode on Modify tab
    elements.manualMode.checked = false;
    elements.autoMode.checked = true;
    container.style.cursor = 'default';

    hoverIn({ index: 0 }, env.context.window.viewer3D, {}, container);
    assert.strictEqual(container.style.cursor, 'pointer', 'HoverIn in Auto mode should set cursor: pointer');

    hoverOut({ index: 0 }, env.context.window.viewer3D, {}, container);
    assert.strictEqual(container.style.cursor, 'default', 'HoverOut in Auto mode should restore cursor: default');

    // 6B: In Manual mode on Modify tab
    elements.manualMode.checked = true;
    elements.autoMode.checked = false;
    container.style.cursor = 'default';

    hoverIn({ index: 0 }, env.context.window.viewer3D, {}, container);
    assert.strictEqual(container.style.cursor, 'pointer', 'HoverIn in Manual mode should set cursor: pointer');

    hoverOut({ index: 0 }, env.context.window.viewer3D, {}, container);
    assert.strictEqual(container.style.cursor, 'default', 'HoverOut in Manual mode should restore cursor: default');

    // 6C: When processing
    elements.executeBtn.classList.add('button-processing');
    container.style.cursor = 'default';

    hoverIn({ index: 0 }, env.context.window.viewer3D, {}, container);
    assert.strictEqual(container.style.cursor, 'default', 'HoverIn while processing must NOT set cursor: pointer');
    elements.executeBtn.classList.remove('button-processing');

    // 6D: When on Insert tab
    elements.tabModify.classList.remove('active');
    elements.tabInsert.classList.add('active');
    container.style.cursor = 'default';

    hoverIn({ index: 0 }, env.context.window.viewer3D, {}, container);
    assert.strictEqual(container.style.cursor, 'default', 'HoverIn on Insert tab must NOT set cursor: pointer');

    console.log('  ✅ Test 6 Passed: Hover cursor interactions verified under all conditions.');
}

console.log('\n🎉 All 6 simulation test suites passed successfully!\n');
