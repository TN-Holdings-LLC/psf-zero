// Usage: node make_poc_docs.js OUT_DIR  (needs the docx npm package)
// Builds the psf-os PoC request form and PoC report template (US Letter, the September documents' look).
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  AlignmentType, Header, Footer, PageNumber,
} = require("docx");

const NAVY = "1B2A4A", RED = "C0392B", GREY = "595959", HINT = "9A9A9A", BOX = "F4F5F7", RULE = "D0D4DA";
const FONT = "Calibri";
const W = 9360; // text width in DXA (Letter, 1" margins)
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NOB = { top: NONE, bottom: NONE, left: NONE, right: NONE };

const t = (text, o = {}) => new TextRun({ text, font: FONT, size: o.size || 21, bold: o.bold, italics: o.italics,
  color: o.color || "222222" });
const p = (runs, o = {}) => new Paragraph({ children: Array.isArray(runs) ? runs : [runs],
  spacing: { before: o.before ?? 60, after: o.after ?? 60 }, alignment: o.align });

function title(main, sub) {
  return [p(t(main, { size: 32, bold: true, color: NAVY }), { before: 0, after: 40 }),
          p(t(sub, { italics: true, color: GREY, size: 20 }), { after: 200 })];
}
function section(num, text) {
  return new Paragraph({ children: [t(num + ". ", { bold: true, color: RED, size: 25 }), t(text, { bold: true, color: NAVY, size: 25 })],
    spacing: { before: 280, after: 120 }, keepNext: true, keepLines: true,
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 4 } } });
}
function hint(text) { return new Paragraph({ children: [t(text, { italics: true, color: GREY, size: 19 })],
  spacing: { before: 60, after: 100 }, keepNext: true }); }
function box(lines) {
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({ children: [
    new TableCell({ width: { size: W, type: WidthType.DXA }, shading: { fill: BOX, type: ShadingType.CLEAR, color: "auto" },
      margins: { top: 140, bottom: 140, left: 200, right: 200 },
      borders: { top: { style: BorderStyle.SINGLE, size: 4, color: RULE }, bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
                 left: { style: BorderStyle.SINGLE, size: 4, color: RULE }, right: { style: BorderStyle.SINGLE, size: 4, color: RULE } },
      children: lines.map((l) => p(Array.isArray(l) ? l : t(l, { size: 20 }), { before: 30, after: 30 })) })] })] });
}
// A label and a write-in line (bottom border under the value cell).
function field(label, example) {
  const lw = 3600, vw = W - lw;
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [lw, vw], rows: [new TableRow({ children: [
    new TableCell({ width: { size: lw, type: WidthType.DXA }, borders: NOB, margins: { top: 80, bottom: 40, left: 0, right: 120 },
      children: [p(t(label, { bold: true, size: 20 }), { before: 20, after: 20 })] }),
    new TableCell({ width: { size: vw, type: WidthType.DXA }, margins: { top: 80, bottom: 40, left: 120, right: 0 },
      borders: { top: NONE, left: NONE, right: NONE, bottom: { style: BorderStyle.SINGLE, size: 4, color: "BBBBBB" } },
      children: [p(t(example || "", { italics: true, color: HINT, size: 19 }), { before: 20, after: 20 })] })] })] });
}
function check(text, note) {
  const runs = [t("☐  ", { size: 22 }), t(text, { size: 21 })];
  if (note) runs.push(t("  (" + note + ")", { italics: true, color: GREY, size: 18 }));
  return p(runs, { before: 40, after: 40 });
}
function label(text) { return p(t(text, { bold: true, size: 21 }), { before: 160, after: 60 }); }
function writeBox(text, rows = 3) {
  const kids = [p(t(text || "", { italics: true, color: GREY, size: 19 }), { before: 40, after: 40 })];
  for (let i = 1; i < rows; i++) kids.push(p(t(""), { before: 40, after: 40 }));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({ cantSplit: true, children: [
    new TableCell({ width: { size: W, type: WidthType.DXA }, margins: { top: 100, bottom: 100, left: 160, right: 160 },
      borders: { top: { style: BorderStyle.SINGLE, size: 4, color: RULE }, bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
                 left: { style: BorderStyle.SINGLE, size: 4, color: RULE }, right: { style: BorderStyle.SINGLE, size: 4, color: RULE } },
      children: kids })] })] });
}
function grid(head, rows, widths) {
  const cell = (txt, i, hdr) => new TableCell({ width: { size: widths[i], type: WidthType.DXA },
    shading: hdr ? { fill: "E8EBF0", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    borders: { top: { style: BorderStyle.SINGLE, size: 4, color: RULE }, bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
               left: { style: BorderStyle.SINGLE, size: 4, color: RULE }, right: { style: BorderStyle.SINGLE, size: 4, color: RULE } },
    children: [p(t(txt, { bold: hdr, size: 19 }), { before: 10, after: 10 })] });
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((h, i) => cell(h, i, true)) }),
           ...rows.map((r) => new TableRow({ children: r.map((x, i) => cell(x, i, false)) }))] });
}
function doc(headerText, children) {
  return new Document({
    styles: { default: { document: { run: { font: FONT, size: 21 } } } },
    sections: [{
      properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
        children: [t(headerText, { size: 16, color: GREY })] })] }) },
      footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: GREY })] })] }) },
      children,
    }],
  });
}

// ------------------------------------------------------------------------------------------- request form
const form = doc("psf-os — PoC Request Form", [
  ...title("PSF-ZERO / PSF-OS — POC REQUEST FORM", "Tell us what you want to run, so we can scope it before starting."),
  box([
    "This form takes about five minutes. It lets us scope an evaluation of psf-os against your own computation before any commitment on either side.",
    "psf-os places a ring circuit on an IBM heavy-hex device without reading any calibration quality number. It tests the device with its own short circuits, avoids the paths it found dead, and reports how far each result can be trusted. Every result is given twice: on the standard placement and on the self-checked one.",
    "If your computation is sensitive, we send a Mutual NDA first. You only need to describe its shape here.",
  ]),
  section("1", "Your organization"),
  field("Company / organization name:"), field("Primary contact (name, role):"), field("Email:"),
  field("How did you find PSF-Zero?", "e.g. paper, GitHub, referral"),

  section("2", "The computation you want to run"),
  hint("Rough shape only at this stage. The exact structure can wait for the NDA, if one is needed."),
  field("Approximate qubit count:"), field("Approximate two-qubit gate count / layers:"),
  label("Shape of the circuit (check any that apply):"),
  check("A ring or a line of qubits (nearest-neighbour two-qubit gates)", "the shape psf-os places today"),
  check("Time evolution of a 1D model (e.g. kicked Ising, Trotterized chain)"),
  check("A layered variational circuit on a ring or a line"),
  check("A benchmark or test circuit with a known answer (e.g. Clifford, mirror)"),
  check("Another shape", "scoped case by case; psf-os does not place it yet"),
  field("If another shape, describe briefly:"),

  section("3", "Device and access"),
  check("A specific IBM device", "name it below"),
  check("Any IBM heavy-hex device is fine"),
  check("Simulation only (a fake device, no IBM account needed)"),
  field("Device name, if known:"),
  label("Who runs the jobs on the device?"),
  check("We run them on our own IBM Quantum account", "we provide the commands"),
  check("We would like TN Holdings to run them", "device time and cost agreed in advance"),
  check("Not decided yet"),

  section("4", "What you want measured"),
  check("Usable width: how many ring sites still carry a signal"),
  check("Readable depth: how many layers the ring stays readable"),
  check("Accuracy of your computation: standard placement against self-checked placement"),
  check("Sentinel verdicts during your own jobs (CLEAN, WEAK, DEAD, with the sites)"),
  check("Device time spent on the self-check (billed seconds)"),
  check("Something else:"),
  field("If something else, describe:"),

  section("5", "Confidentiality"),
  check("Our computation is not sensitive — no NDA needed to proceed"),
  check("Our computation is sensitive — please send the Mutual NDA first"),
  check("Not sure yet — let's discuss"),

  section("6", "Anything else we should know?"),
  writeBox("", 4),
  p(t(""), { after: 120 }),
  p([t("Send this completed form to love.os.architect@proton.me. We confirm the scope, send a Mutual NDA if you asked for one, and fix the test in writing (what is run, what is measured, what would count as a result) before anything is run.",
    { italics: true, color: GREY, size: 19 })]),
]);

// ------------------------------------------------------------------------------------------- report template
const report = doc("psf-os — PoC Report", [
  ...title("PSF-ZERO / PSF-OS — POC REPORT", "Standard placement and self-checked placement, side by side."),
  field("Prepared for:"), field("Date (CET):"), field("Prepared by:", "TN Holdings"),

  section("1", "Summary"),
  writeBox("Two or three sentences. Give the headline number and, next to it, the essential number: what the result means for someone running this computation on the device. A reader who stops here should know what happened.", 3),

  section("2", "The computation"),
  hint("What was run, in enough detail to repeat it, without disclosing confidential content."),
  field("Qubit count / ring length:"), field("Two-qubit gate count / layers:"), field("Shape of the circuit:"),
  field("Known answer (how it was computed):", "e.g. stabilizer simulation, state vector"),
  field("Device(s) and run dates (CET):"),

  section("3", "Declaration (fixed before any run)"),
  hint("The test was written down and fixed before the day's device data was fetched. Changes after that are listed here as amendments."),
  field("Declaration file:"), field("SHA-256:"), field("Fixed at (CET):"), field("Amendments, if any:"),

  section("4", "Method"),
  hint("What was measured, and how, so the numbers in Section 6 can be checked, not just trusted."),
  field("psf-os / TOQB versions (commits):"),
  field("Device data used:", "coupling graph and failed elements only; no quality number"),
  field("Scan jobs (ids, billed seconds):"),
  field("Defect-list entries applied:", "paths, the dates they were seen"),
  field("Shots, options:", "e.g. 1000 shots; dynamical decoupling and twirling off"),
  field("Job ids, billed seconds:"),

  section("5", "Correctness"),
  writeBox("How the compiled circuits were checked to compute the same thing as the logical circuit (e.g. noiseless simulation returns the known answer; operator equality where the size allows), and at what scale.", 3),

  section("6", "Results"),
  grid(["Measure", "Standard placement", "Self-checked placement", "Difference (± 2 SE)"], [
    ["Usable sites (of n)", "", "", ""],
    ["Mean signal on usable sites", "", "", ""],
    ["Score of the computation (state the measure)", "", "", ""],
    ["Sentinel verdict (sites)", "", "", ""],
    ["Device time, billed seconds", "", "", ""],
  ], [3000, 2120, 2120, 2120]),
  p(t(""), { after: 60 }),
  writeBox("Headline number and essential number, side by side. Say which figures are single runs and which are averages, and where the difference comes from (e.g. only the replaced qubits). Report results that did not favour psf-os in the same way as those that did.", 3),

  section("7", "What the self-check found"),
  writeBox("Dead and weak stretches found by the scan (paths, dates), what was avoided, sentinel findings during the jobs, and anything that came and went within the day.", 3),

  section("8", "Limitations"),
  writeBox("What this PoC does not establish: devices, days and circuit shapes not tested; sizes not reached; anything reduced from the published method to fit the PoC's time.", 3),

  section("9", "Recommended next step"),
  writeBox("One of: a deeper evaluation; a commercial conversation (settled in a separate agreement, not in this report); or this computation is not a good fit today, and why.", 3),
]);

const out = process.argv[2];
Promise.all([Packer.toBuffer(form), Packer.toBuffer(report)]).then(([a, b]) => {
  fs.writeFileSync(out + "/psf_os_poc_request_form.docx", a);
  fs.writeFileSync(out + "/psf_os_poc_report_template.docx", b);
  console.log("written");
});
