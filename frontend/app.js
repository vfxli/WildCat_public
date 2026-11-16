import { createApp, ref, reactive, computed, onMounted, onBeforeUnmount, nextTick, watch, provide, inject } from '/static/vendor/vue.esm.js';
import katex from '/static/vendor/katex/katex.esm.js';

const AUTO_BATCH = { Small: 8, Medium: 8, Large: 8 }; // CopyCat 'Auto' batch = 8 (memory permitting)

const ADV_LATEX = {
  cg_grid: String.raw`g_s = \max\!\left(1,\; \left\lfloor \frac{g_{\mathrm{full}}}{s} \right\rfloor\right), \qquad x_0 = g_s \cdot k_x, \qquad y_0 = (H_s - C_s) - g_s \cdot k_y`,
  cg_uniform: String.raw`g_{\mathrm{full}}^{\mathrm{Nuke}} = 4:\;\; y_0 \equiv (H - C) \pmod{4} \;\;\bigl(H \bmod 4 \neq 0 \implies \phi_{\mathrm{train}} \neq \phi_{\mathrm{infer}}\bigr), \qquad g_{\mathrm{full}} = 1:\;\; (x_0 \bmod 4,\, y_0 \bmod 4) \sim \mathcal{U}\bigl(\{0,1,2,3\}^2\bigr)`,
  lf_hdr: String.raw`d(y) = \bigl(1 + \operatorname{Lum}(|y|)\bigr)^{\alpha}, \qquad r_{\alpha}(\hat{y}, y) = \frac{\hat{y} - y}{d(y)}, \qquad \mathcal{L}_{\mathrm{Charb}}(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N} \sqrt{r_{\alpha,i}^2 + \epsilon^2}`,
  lf_total: String.raw`\mathcal{L}_{\mathrm{comp}} = (1 - \lambda_{\mathrm{charb}})\cdot \frac{1}{N}\left\|r_{\alpha}\right\|_2^2 + \lambda_{\mathrm{charb}}\,\mathcal{L}_{\mathrm{Charb}} + \lambda_{\mathrm{grad}}\cdot \Bigl(\left\|\widetilde{\Delta}_x(\hat{y} - y)\right\|_1 + \left\|\widetilde{\Delta}_y(\hat{y} - y)\right\|_1\Bigr)`,
  st_nuke: String.raw`T_{\mathrm{stage}}^{\mathrm{Nuke}} = \left(\left\lceil\frac{T}{3}\right\rceil,\; \min\!\left(T,\,2\left\lceil\frac{T}{3}\right\rceil\right),\; T\right), \qquad s:\; \frac{1}{4}\;\longrightarrow\;\frac{1}{2}\;\longrightarrow\;1`,
  st_custom: String.raw`T_{1/4} = \operatorname{round}(r_{1/4}\,T),\qquad T_{1/2} = \operatorname{round}\bigl((r_{1/4}+r_{1/2})\,T\bigr),\qquad T_{1/1} = T \qquad (r_{1/4} + r_{1/2} + r_{1/1} = 1)`,
  st_warmup: String.raw`\eta_{\mathrm{eff}}(k) = \eta_{\mathrm{Nuke}}(k)\cdot \min\!\left(1,\; 0.1 + 0.9\,\frac{k - T_{\mathrm{start}}}{\max(1,\, K_{\mathrm{warm}})}\right), \qquad \eta_{\mathrm{Nuke}}(k) = \eta_0\!\left(1 - \frac{k-1}{T}\right)^{0.3}`,
  aug_formula: String.raw`\Delta e \sim \mathcal{U}(-E_{\max},\, +E_{\max}), \quad \kappa = 2^{\Delta e}, \quad \mathcal{G}_{\kappa}(u)_{c} = \begin{cases} \kappa\, u_{c}, & c \in \{R,G,B\}\; (C \ge 3) \\[3pt] u_{c}, & c = A \;\text{or}\; C = 1 \end{cases}, \quad (\tilde{x}, \tilde{y}) = \bigl(\mathcal{G}_{\kappa}(\mathcal{F}_{p}(x)),\; \mathcal{G}_{\kappa}(\mathcal{F}_{p}(y))\bigr)`,
};

function renderKatexHtml(tex, displayMode = true) {
  try {
    return katex.renderToString(tex, { displayMode, throwOnError: false, strict: false });
  } catch (e) {
    return `<code>${tex}</code>`;
  }
}

const ADV_MATH_HTML = Object.fromEntries(
  Object.entries(ADV_LATEX).map(([k, v]) => [k, renderKatexHtml(v, true)])
);

const PLACEHOLDER_RE = /\{([a-zA-Z0-9_]+)\}/g;

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function lookupLocalePath(bundle, key) {
  if (!bundle || !key) return undefined;
  let cur = bundle;
  for (const part of String(key).split('.')) {
    if (cur && typeof cur === 'object' && part in cur) {
      cur = cur[part];
    } else {
      return undefined;
    }
  }
  return typeof cur === 'string' ? cur : undefined;
}

// like lookupLocalePath, but returns objects too (the doc.<id> / evidence.<id> entries)
function lookupLocaleNode(bundle, key) {
  let cur = bundle;
  for (const part of String(key || '').split('.')) {
    if (cur && typeof cur === 'object' && part in cur) cur = cur[part];
    else return undefined;
  }
  return cur;
}

function interpolateTemplate(tpl, vars, escapeValues = false) {
  if (!vars || typeof tpl !== 'string') return tpl;
  return tpl.replace(PLACEHOLDER_RE, (m, k) => {
    if (!(k in vars) || vars[k] === undefined || vars[k] === null) return m;
    return escapeValues ? escapeHtml(vars[k]) : String(vars[k]);
  });
}

function applyHtmlLang(bundle, fallbackCode) {
  if (typeof document !== 'undefined' && document.documentElement) {
    const htmlLang = (bundle && bundle._meta && bundle._meta.htmlLang) || fallbackCode || 'zh-CN';
    document.documentElement.lang = htmlLang;
  }
}

// Ranges and defaults of advanced.kf_common - the same as wildcat/advanced.py (section 06)
const KF_COMMON_LIMITS = Object.freeze({
  threshold: Object.freeze({ min: 0.5, max: 16, def: 2.0 }),
  span: Object.freeze({ min: 1, max: 4, def: 2 }),
});

function createDefaultAdvConfig() {
  return {
    // 01. Crop origin sub-pixel phase grid (eliminates Nuke 4px bottom-anchor phase mismatch)
    crop_grid: {
      enabled: false,
      full_res_grid: 1, // 1 | 2 | 4 (Nuke native = 4)
      anchor: 'bottom', // 'bottom' (CopyCat) | 'top' (aligned with Inference); only meaningful for grid 2 / 4
    },
    // 02. Composite loss: Charbonnier + L1 Spatial Edge Gradient + ACEScg HDR Highlight Damping.
    //     Values offered when the module is switched on by hand follow the 2026-11 re-validation: only the gradient term
    //     showed a gain (keying), Charbonnier lowered unseen-frame PSNR and HDR damping had no measurable effect.
    loss_family: {
      enabled: false,
      charbonnier_ratio: 0,
      charb_eps: 0.001,
      grad_weight: 0.05,
      hdr_alpha: 0,
    },
    // 03. Multi-resolution stage duration split & stage transition LR soft warmup
    stage_schedule: {
      enabled: false,
      r_quarter: 15,
      r_half: 25,
      r_full: 60,
      warmup_steps: 200,
    },
    // 04. Averaged weights (EMA shadow), off by default. Restored 2026-11 (Patch 43): a finished run comes out the same, but
    //     checkpoints taken before the end are steadier (old campaign: 1,827 checkpoints; re-validation; face shots).
    ema_lr: {
      enabled: false,
      ema_decay: 0.9995,
    },
    // 05. Semantic-preserving optical augmentation (HFlip=0.5, EV=0.0). Not part of any preset: the re-validation found no
    //     consistent effect of the horizontal flip (one clean-up shot).
    augment: {
      enabled: false,
      hflip_prob: 0.5,
      exposure_ev: 0.0,
    },
    // 06 (shown as 05). Keyframe commonality: before training, a Ground Truth keyframe is rewritten where the other
    //     keyframes agree with each other and this one does not (wildcat/kfcommon.py). RGB -> RGB only; needs the plate as
    //     the dataset's evaluation sequence and at least 3 keyframes. A one-off preparation step, no per-step cost.
    kf_common: {
      enabled: false,
      threshold: KF_COMMON_LIMITS.threshold.def, // smaller = rewrites more
      span: KF_COMMON_LIMITS.span.def, // keyframes looked at on each side
      flow: 'dis', // optical flow the keyframes are compared with: 'dis' (default) | 'raft' (offered where the server can run it)
      overall: true, // overall look (broad strength / colour) also pulled to the keyframes' consensus where it disagrees
    },
  };
}

// threshold / span / flow as the trainer will use them (same rules as parse_advanced_config; threshold kept to 3 decimals)
function clampKfCommon(kc) {
  const lim = KF_COMMON_LIMITS;
  const num = (v) => (v === null || v === undefined || v === '' ? NaN : Number(v)); // an emptied field = the default
  const thr = Math.round(num(kc && kc.threshold) * 1000) / 1000;
  const span = num(kc && kc.span);
  return {
    threshold: Number.isFinite(thr) ? Math.min(lim.threshold.max, Math.max(lim.threshold.min, thr)) : lim.threshold.def,
    span: Number.isFinite(span) ? Math.trunc(Math.min(lim.span.max, Math.max(lim.span.min, span))) : lim.span.def,
    flow: String((kc && kc.flow) || '').trim().toLowerCase() === 'raft' ? 'raft' : 'dis',
    overall: !(kc && (kc.overall === false || kc.overall === 0 || kc.overall === 'false')), // missing = on, as in wildcat/advanced.py
  };
}

// The face-edit preset, written down once: the preset button, the help recipe and the anchor of the combo test read it.
// (1px crop grid on like every preset; stage schedule and augmentation off; Epochs untouched; the optical flow stays the
// default DIS - RAFT is a choice the user makes in the card.) Model Size Medium = the CopyCat default: on the public face
// shots Medium beats Small by 0.9-2.6 dB with and without keyframe commonality (Patch 50). Loss: Charbonnier 0.25 with
// no gradient term and no highlight damping (Patch 56: better on every face shot, seed and keyframe condition tested -
// unseen frames, edited area, and far fewer changes to the pixels the edit does not touch; evidence.preset_face_edit).
const FACE_EDIT_PRESET = Object.freeze({
  model_size: 'Medium',
  channels: '3->3',
  crop: 256,
  loss: Object.freeze({ charbonnier_ratio: 0.25, grad_weight: 0, hdr_alpha: 0 }),
  kf_common: Object.freeze({ threshold: 2.0, span: 2, flow: 'dis', overall: true }),
  ema_lr: Object.freeze({ enabled: true, ema_decay: 0.9995 }),
});
// Averaged weights (EMA) in the presets of the task types where checkpoints taken before the end gain from them
// (keying, clean-up, face edits; Patch 43 + the combined judgment of both test campaigns). A run trained to the end
// comes out the same; defocus, degrain and roto showed no difference, so their presets leave it off.
const PRESET_EMA = Object.freeze({ enabled: true, ema_decay: 0.9995 });

// Scene presets, written down once: applyEmpiricalPresetCore() applies a spec, the help center lists it (labels from the
// parameter reference). Every preset switches Multi-Resolution on, uses the 1-px crop grid and leaves Epochs alone;
// anything not named here is switched off. hdr_alpha is forced to 0 for a 1-channel output.
const PRESET_IDS = ['nuke', 'cleanup_grain', 'keying_despill', 'bokeh_focus', 'degrain_clean', 'roto_matte', 'face_edit'];
const PRESET_SPECS = Object.freeze({
  // Medium (the CopyCat default): +1.1 dB over Small on the clean-up test shot (Patch 50); 15 / 25 / 60 +0.64 dB on Medium
  // with averaged weights, every seed higher, about 1.33 times the training time (Patch 58)
  cleanup_grain: {
    model_size: 'Medium', channels: '3->3', crop: 256, crop_grid: 1, loss: { charbonnier_ratio: 0, grad_weight: 0.05, hdr_alpha: 0 },
    stages: { r_quarter: 15, r_half: 25, r_full: 60, warmup_steps: 200 }, ema: PRESET_EMA,
  },
  keying_despill: { model_size: 'Small', channels: '3->4', crop: 256, crop_grid: 1, loss: { charbonnier_ratio: 0, grad_weight: 0.15, hdr_alpha: 0 }, ema: PRESET_EMA },
  bokeh_focus: {
    model_size: 'Large', channels: '3->3', crop: 256, crop_grid: 1,
    loss: { charbonnier_ratio: 0.25, grad_weight: 0.1, hdr_alpha: 0.5 },
    stages: { r_quarter: 15, r_half: 25, r_full: 60, warmup_steps: 200 },
  },
  degrain_clean: { model_size: 'Small', channels: '3->3', crop: 256, crop_grid: 1 },
  roto_matte: { model_size: 'Small', channels: '3->1', crop: 256, crop_grid: 1 },
  face_edit: {
    model_size: FACE_EDIT_PRESET.model_size, channels: FACE_EDIT_PRESET.channels, crop: FACE_EDIT_PRESET.crop, crop_grid: 1,
    loss: FACE_EDIT_PRESET.loss, kf_common: FACE_EDIT_PRESET.kf_common, ema: FACE_EDIT_PRESET.ema_lr,
  },
});

// ───────── Parameter reference and strategy evidence ─────────
// One source for every explanation on the page: the locale entries doc.<id> {title, summary, body} (parameter reference)
// and evidence.<id> {title, summary, body} (test results, advanced help tab; <id> = the parameter it supports).
// The "?" control (<help-tip doc="id">: title + summary on hover / focus, click = help center at the entry), the help
// center and the cards of the advanced panel all render these keys. Private-edition entries live in
// frontend/locales/private/*.json and are listed in the private-edition block below.
// Body markup (renderDocMarkup): blank line = new block, single newline = line break; "## " line = sub-heading;
// "- " lines = list; "|a|b|" lines = table (first row header, "|---|" row skipped); inline **bold**, `code`,
// [[doc:id]] / [[ev:id]] (optional "|label") = link to another entry. Raw HTML is escaped.
const DOC_GROUPS = [
  { id: 'train', ids: ['task_name', 'dataset', 'device', 'init_weights', 'model_size', 'epochs', 'batch_size', 'crop_size', 'multi_resolution', 'mixed_precision', 'channels', 'checkpoint_interval', 'contact_sheet_interval', 'learning_rate', 'seed', 'log_interval', 'color_pipeline', 'model_files', 'speed_options', 'speed_gpu_resident', 'speed_cuda_graph', 'speed_fused_adam', 'speed_channels_last', 'speed_cudnn_benchmark', 'speed_async_io', 'preview_gpu', 'eta_estimate'] },
  { id: 'adv', ids: ['adv_overview', 'adv_crop_grid', 'adv_crop_grid_step', 'adv_crop_grid_anchor', 'adv_loss', 'adv_loss_charbonnier', 'adv_loss_grad', 'adv_loss_hdr', 'adv_loss_eps', 'adv_stages', 'adv_stages_ratio', 'adv_stages_warmup', 'adv_ema', 'adv_ema_decay', 'adv_augment', 'adv_augment_flip', 'adv_augment_exposure', 'adv_kf_common', 'adv_kf_common_threshold', 'adv_kf_common_span', 'adv_kf_common_overall'] },
  { id: 'presets', ids: ['presets_overview', 'preset_nuke', 'preset_cleanup_grain', 'preset_keying_despill', 'preset_bokeh_focus', 'preset_degrain_clean', 'preset_roto_matte', 'preset_face_edit'] },
  { id: 'lab', ids: ['lab_overview', 'lab_combo', 'lab_combo_dataset', 'lab_combo_task_type', 'lab_combo_scope', 'lab_group_name', 'lab_native_baseline', 'lab_note', 'lab_staged_groups', 'lab_compare_tasks', 'lab_compare_baseline', 'lab_param_tags', 'lab_eval', 'lab_eval_dataset', 'lab_eval_weights', 'lab_chart', 'lab_viewer', 'lab_db', 'lab_archive', 'metric_psnr_srgb', 'metric_edge', 'metric_bg_drift', 'metric_p99', 'metric_seq_delta', 'metric_hfi', 'metric_static_boil', 'metric_seq_psnr'] },
  { id: 'datasets', ids: ['dataset_overview', 'dataset_keyframes', 'dataset_validation_sequence', 'dataset_task_type', 'dataset_input_colorspace', 'dataset_check', 'dataset_tolerance', 'dataset_files', 'dataset_name', 'dataset_eval_sequence', 'dataset_demo'] },
  { id: 'monitor', ids: ['monitor_pause_resume', 'monitor_extend', 'monitor_download', 'monitor_delete', 'monitor_progress', 'monitor_stage', 'monitor_speed', 'monitor_eta', 'monitor_queue', 'metric_loss', 'metric_val_mse', 'monitor_failed', 'monitor_kfc', 'metric_kfc_rewritten', 'metric_kfc_common', 'monitor_preview', 'monitor_preview_keyframe', 'monitor_playback', 'monitor_contact_sheet', 'monitor_cs_overlay', 'monitor_cs_pixelated', 'monitor_loss_chart', 'monitor_loss_log', 'monitor_loss_smooth', 'monitor_quality', 'metric_kf_fit', 'metric_q_strength', 'metric_q_detail', 'metric_q_texture', 'metric_q_look_drift', 'metric_q_tex_drift', 'metric_q_bg', 'monitor_quality_chart', 'monitor_quality_strip', 'monitor_quality_table', 'monitor_models', 'monitor_model_file', 'monitor_model_tags', 'monitor_log'] },
  { id: 'tools', ids: ['api_key'] },
  { id: 'admin', ids: ['admin_max_jobs_per_gpu', 'admin_gpu_pool', 'admin_temp_restrictions', 'admin_default_quotas', 'admin_max_running_per_user', 'admin_users', 'admin_user_quota', 'admin_user_permissions', 'admin_storage_root', 'admin_analytics_db'] },
];
const EVIDENCE_GROUPS = [
  { id: 'method', ids: ['method', 'appendix_material'] },
  { id: 'adv', ids: ['adv_crop_grid', 'adv_crop_grid_anchor', 'adv_loss_grad', 'adv_loss_charbonnier', 'adv_loss_hdr', 'adv_stages', 'adv_ema', 'adv_augment_flip', 'adv_kf_common'] },
  { id: 'training', ids: ['model_size', 'crop_size', 'epochs', 'dataset_keyframes', 'speed_options'] },
  { id: 'presets', ids: ['preset_cleanup_grain', 'preset_keying_despill', 'preset_bokeh_focus', 'preset_degrain_clean', 'preset_roto_matte', 'preset_face_edit'] },
];

// markup of a doc / evidence body -> HTML. resolve(kind, id) gives the label of a link target ('' = no such entry).
function renderDocMarkup(src, resolve) {
  const inline = (s) => {
    let h = escapeHtml(s);
    h = h.replace(/\[\[(doc|ev):([a-z0-9_]+)(?:\|([^\]]*))?\]\]/g, (m, kind, id, label) => {
      const target = resolve(kind, id);
      if (!target) return label || id;
      const attr = kind === 'doc' ? 'data-doc' : 'data-ev';
      return `<a href="#${kind}-${id}" class="wc-doc-link" ${attr}="${id}">${label || escapeHtml(target)}</a>`;
    });
    h = h.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
    return h;
  };
  const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let para = [];
  let list = [];
  let rows = [];
  const flush = () => {
    if (para.length) out.push(`<p>${para.map(inline).join('<br>')}</p>`);
    if (list.length) out.push(`<ul>${list.map((x) => `<li>${inline(x)}</li>`).join('')}</ul>`);
    if (rows.length) {
      const cells = rows
        .filter((r) => !/^\|[\s:|-]+\|$/.test(r))
        .map((r) => r.replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => inline(c.trim())));
      if (cells.length) {
        const [head, ...body] = cells;
        out.push(
          '<div class="wc-doc-table"><table><thead><tr>' + head.map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
          body.map((r) => '<tr>' + r.map((c) => `<td>${c}</td>`).join('') + '</tr>').join('') + '</tbody></table></div>'
        );
      }
    }
    para = [];
    list = [];
    rows = [];
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      flush();
    } else if (/^##\s+/.test(line)) {
      flush();
      out.push(`<h5>${inline(line.replace(/^##\s+/, ''))}</h5>`);
    } else if (/^-\s+/.test(line)) {
      if (para.length || rows.length) flush();
      list.push(line.replace(/^-\s+/, ''));
    } else if (/^\|.*\|$/.test(line)) {
      if (para.length || list.length) flush();
      rows.push(line);
    } else {
      if (list.length || rows.length) flush();
      para.push(line);
    }
  }
  flush();
  return out.join('');
}

// The "?" next to a parameter or an indicator: hover / keyboard focus shows title + summary of doc.<id> in a popover
// (teleported to <body>, so no scroll container clips it), click opens the help center at the entry.
let helpTipSeq = 0;
const HelpTip = {
  name: 'HelpTip',
  props: { doc: { type: String, required: true } },
  setup(props) {
    const wc = inject('wcHelp', null);
    const open = ref(false);
    const btn = ref(null);
    const pop = reactive({ left: 0, top: 0, width: 300, below: true });
    const tipId = `wc-help-pop-${++helpTipSeq}`;
    let timer = 0;
    const title = computed(() => (wc ? wc.docTitle(props.doc) : props.doc));
    const summary = computed(() => (wc ? wc.docSummary(props.doc) : ''));
    const more = computed(() => (wc ? wc.t('help.tipMore') : ''));
    const label = computed(() => (wc ? wc.t('help.tipAria', { title: title.value }) : title.value));
    const onViewportChange = () => hide();
    function place() {
      const el = btn.value;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const vw = window.innerWidth || document.documentElement.clientWidth || 1024;
      const vh = window.innerHeight || document.documentElement.clientHeight || 768;
      const w = Math.min(320, vw - 24);
      pop.width = w;
      pop.left = Math.round(Math.min(Math.max(12, r.left + r.width / 2 - w / 2), vw - w - 12));
      pop.below = r.bottom + 170 < vh || r.top < 170;
      pop.top = Math.round(pop.below ? r.bottom + 8 : r.top - 8);
    }
    function show() {
      clearTimeout(timer);
      place();
      if (!open.value) {
        open.value = true;
        window.addEventListener('scroll', onViewportChange, true);
        window.addEventListener('resize', onViewportChange);
      }
    }
    function hide() {
      clearTimeout(timer);
      if (open.value) {
        open.value = false;
        window.removeEventListener('scroll', onViewportChange, true);
        window.removeEventListener('resize', onViewportChange);
      }
    }
    function enter() {
      clearTimeout(timer);
      timer = setTimeout(show, 120);
    }
    function onClick(ev) {
      ev.preventDefault();
      ev.stopPropagation();
      hide();
      if (wc) wc.openDoc(props.doc, btn.value);
    }
    function onKey(ev) {
      if (ev.key === 'Escape' && open.value) {
        ev.preventDefault();
        ev.stopPropagation();
        hide();
      }
    }
    onBeforeUnmount(hide);
    return { open, btn, pop, tipId, title, summary, more, label, enter, hide, show, onClick, onKey };
  },
  template:
    '<span class="wc-help-wrap"><button ref="btn" type="button" class="wc-help-tip" :data-help-doc="doc" :aria-label="label"' +
    ' :aria-describedby="open ? tipId : undefined" @mouseenter="enter" @mouseleave="hide" @focus="show" @blur="hide"' +
    ' @click="onClick" @keydown="onKey">?</button><teleport to="body"><div v-if="open" :id="tipId" role="tooltip"' +
    ' class="wc-help-pop" :class="pop.below ? \'is-below\' : \'is-above\'"' +
    ' :style="{ left: pop.left + \'px\', top: pop.top + \'px\', width: pop.width + \'px\' }">' +
    '<div class="wc-help-pop-title">{{ title }}</div><div v-if="summary" class="wc-help-pop-sum">{{ summary }}</div>' +
    '<div class="wc-help-pop-more">{{ more }}</div></div></teleport></span>',
};

// Why keyframe commonality would not take effect for a model / dataset: '' | 'channels' | 'no_plate' | 'few_keyframes'.
// Only tells the user beforehand - the trainer makes the final call, trains as usual and reports the reason.
function kfCommonBlocker(cIn, cOut, ds) {
  if (Number(cIn) !== 3 || Number(cOut) !== 3) return 'channels';
  if (!ds) return '';
  if (!ds.has_eval_sequence) return 'no_plate';
  if (Number(ds.frame_count || 0) > 0 && Number(ds.frame_count) < 3) return 'few_keyframes';
  return '';
}

// ───────── Training quality on the in-between frames (telemetry field "quality") ─────────
// The five charted measures. Colours: the first five slots of the chart palette in their fixed order, checked as a set
// on the chart surface #020617 (lightness band, chroma floor, separation of neighbouring slots under colour-vision
// deficiency, contrast). A measure keeps its colour everywhere: tile key, key line of its plot, curve, per-frame bars.
// plot 0 = how much of the edit arrives on the in-between frames (1.0 = as on the keyframes), plot 1 = how unsteady it
// is over time (lower = steadier), plot 2 = fit of the training keyframes in dB (its own scale). Never more than two
// curves share a plot: curves cross, and four of these colours cannot all be told apart from each other. The two curves
// of a plot also differ by their marker (circle / square), so telling them apart does not depend on colour.
const QUALITY_SERIES = Object.freeze([
  Object.freeze({ id: 'strength', field: 'strength_mean', plot: 0, color: '#3987e5', marker: 'circle', digits: 2 }),
  Object.freeze({ id: 'detail', field: 'detail', plot: 0, color: '#d95926', marker: 'square', digits: 2 }),
  Object.freeze({ id: 'look', field: 'look_drift5', plot: 1, color: '#199e70', marker: 'circle', digits: 2 }),
  Object.freeze({ id: 'tex', field: 'tex_drift1', plot: 1, color: '#c98500', marker: 'square', digits: 2 }),
  Object.freeze({ id: 'fit', field: 'kf_fit_roi_db', plot: 2, color: '#d55181', marker: 'circle', digits: 1 }),
]);
const QUALITY_POINT_FIELDS = Object.freeze([
  'seconds', 'kf_fit_db', 'kf_fit_roi_db', 'strength_mean', 'strength_min', 'strength_min_frame',
  'detail', 'texture', 'look_drift5', 'tex_drift1', 'tex_drift5', 'bg_change',
]);
const QUALITY_AXIS_MAX = 1.2; // fixed top of the 0..1.2 axes (strength, detail, drift): the same scale for every run
const QUALITY_HINT_BELOW = 0.6; // the two plain hints of the card (edit fades on a frame, detail is soft) appear below this
// per-frame bars: the colour of "strength"; the weakest frame is set apart by lightness (and by its label), not by a
// warning colour - there is no validated threshold that would justify one
const QUALITY_BAR = Object.freeze({ normal: '#3987e5', hover: '#6da7ec', weakest: '#cde2fb' });
// the parameter-reference entry (doc.<id>) that explains each measure of the card: the "?" of its tile and table column
const QUALITY_DOC = Object.freeze({
  fit: 'metric_kf_fit', strength: 'metric_q_strength', detail: 'metric_q_detail', look: 'metric_q_look_drift', tex: 'metric_q_tex_drift', bg: 'metric_q_bg',
});

// a finite number, or null for anything else (a report carries null for a value that could not be measured)
function qualityNum(v) {
  if (v === null || v === undefined || v === '' || typeof v === 'boolean') return null;
  const x = Number(v);
  return Number.isFinite(x) ? x : null;
}

function qualityFmt(v, digits = 2) {
  const x = qualityNum(v);
  return x === null ? '—' : x.toFixed(digits);
}

// The ratios of the card (strength, detail, drift, background change) are shown as percentages: 0.56 -> "56%".
// strength / detail: 100% = as on the keyframes; drift: the change as a share of the band's own energy (lower = steadier);
// background change: mean change of a pixel outside the edited area, as a share of full white (1.0 = 100%).
function qualityPct(v, digits = 0) {
  const x = qualityNum(v);
  return x === null ? '—' : (100 * x).toFixed(digits) + '%';
}
// how a charted series reads: fit in dB, everything else in percent
function qualitySeriesText(s, v) {
  if (qualityNum(v) === null) return '—';
  return s.id === 'fit' ? qualityFmt(v, s.digits) + ' dB' : qualityPct(v);
}

// one measurement with every field a number or null; null when it has no step
function qualityPoint(p) {
  if (!p || typeof p !== 'object') return null;
  const step = qualityNum(p.step);
  if (step === null) return null;
  const out = { step };
  for (const f of QUALITY_POINT_FIELDS) out[f] = qualityNum(p[f]);
  return out;
}

// the measurements of a report, oldest first (a step measured twice, e.g. after a resume, counts once: the later one)
function qualityPoints(q) {
  const byStep = new Map();
  for (const p of Array.isArray(q && q.points) ? q.points : []) {
    const n = qualityPoint(p);
    if (n) byStep.set(n.step, n);
  }
  return [...byStep.values()].sort((a, b) => a.step - b.step);
}

// Per-frame strip of the latest measurement: one bar per evaluated in-between frame, placed by its plate frame number,
// the keyframes marked in the gaps. All positions are percentages of the strip's width / height.
function qualityStrip(latestRaw, header) {
  const pf = latestRaw && latestRaw.per_frame && typeof latestRaw.per_frame === 'object' ? latestRaw.per_frame : null;
  const fr = Array.isArray(pf && pf.frame) ? pf.frame : [];
  const st = Array.isArray(pf && pf.strength) ? pf.strength : [];
  const seen = new Set();
  let bars = [];
  for (let i = 0; i < fr.length; i++) {
    const frame = qualityNum(fr[i]);
    if (frame === null || seen.has(frame)) continue;
    seen.add(frame);
    bars.push({ frame, value: qualityNum(st[i]) });
  }
  if (!bars.length) return null;
  bars.sort((a, b) => a.frame - b.frame);
  const keyFrames = [...new Set((Array.isArray(header && header.keyframes) ? header.keyframes : []).map(qualityNum).filter((x) => x !== null))]
    .sort((a, b) => a - b);
  const minF = Math.min(bars[0].frame, keyFrames.length ? keyFrames[0] : Infinity);
  const maxF = Math.max(bars[bars.length - 1].frame, keyFrames.length ? keyFrames[keyFrames.length - 1] : -Infinity);
  const span = maxF - minF + 1;
  // spacing of the evaluated frames: header.stride, else the smallest gap between two of them
  let stride = qualityNum(header && header.stride);
  if (!(stride >= 1)) {
    stride = Infinity;
    for (let i = 1; i < bars.length; i++) stride = Math.min(stride, bars[i].frame - bars[i - 1].frame);
    if (!Number.isFinite(stride)) stride = 1;
  }
  const slot = (100 * Math.max(1, stride)) / span;
  const center = (f) => (100 * (f - minF + 0.5)) / span;

  const measured = bars.filter((b) => b.value !== null);
  const wantFrame = qualityNum(latestRaw && latestRaw.strength_min_frame);
  let weakest = wantFrame !== null ? measured.find((b) => b.frame === wantFrame) || null : null;
  if (!weakest && measured.length) weakest = measured.reduce((a, b) => (b.value < a.value ? b : a));

  bars = bars.map((b) => ({
    frame: b.frame,
    value: b.value,
    text: qualityPct(b.value),
    center: center(b.frame),
    left: center(b.frame) - slot / 2,
    width: slot,
    h: b.value === null ? 0 : (100 * Math.min(QUALITY_AXIS_MAX, Math.max(0, b.value))) / QUALITY_AXIS_MAX,
    over: b.value !== null && b.value > QUALITY_AXIS_MAX,
    weakest: !!weakest && b.frame === weakest.frame,
  }));
  // keyframe numbers under the strip: left out where they would run into the previous one
  let lastLabel = -Infinity;
  const keys = keyFrames.map((f) => {
    const c = center(f);
    const label = c - lastLabel >= 4.5;
    if (label) lastLabel = c;
    return { frame: f, center: c, label };
  });
  return {
    bars,
    keys,
    weakest: bars.find((b) => b.weakest) || null,
    over: bars.some((b) => b.over),
    yTicks: [0, 0.5, 1].map((v) => ({ label: qualityPct(v), pct: (100 * v) / QUALITY_AXIS_MAX, ref: v === 1 })),
  };
}

async function loadInitialI18nState() {
  const boot = globalThis.__WC_BOOT_I18N__;
  if (boot && boot.bundles) {
    const locales = Array.isArray(boot.locales) ? boot.locales : [];
    const defCode = boot.default || (locales[0] && locales[0].code) || 'zh-CN';
    const saved = boot.initialLang || (typeof localStorage !== 'undefined' ? localStorage.getItem('wc_lang') : '') || '';
    const validCodes = locales.map((x) => x.code);
    const initCode = validCodes.includes(saved) ? saved : defCode;
    const bundle = boot.bundles[initCode] || boot.bundles[defCode] || {};
    applyHtmlLang(bundle, initCode);
    return {
      defaultCode: defCode,
      currentCode: initCode,
      locales,
      cache: Object.assign({}, boot.bundles),
      bundle,
    };
  }

  let defaultCode = 'zh-CN';
  let locales = [];
  const saved = (typeof localStorage !== 'undefined' ? localStorage.getItem('wc_lang') : '') || '';
  const initialCandidate = saved || defaultCode;

  let resolvedCode = initialCandidate;
  let bundle = {};
  const cache = {};

  try {
    const [catRes, bRes] = await Promise.all([
      fetch('/api/v1/locales', { cache: 'no-cache' }),
      fetch(`/api/v1/locales/${encodeURIComponent(initialCandidate)}`, { cache: 'no-cache' }),
    ]);
    if (catRes.ok) {
      const cat = await catRes.json();
      defaultCode = cat.default || defaultCode;
      locales = Array.isArray(cat.locales) ? cat.locales : [];
    }
    const codes = locales.map((x) => x.code);
    const chosen = saved && (!codes.length || codes.includes(saved)) ? saved : defaultCode;

    if (chosen === initialCandidate && bRes.ok) {
      const bJson = await bRes.json();
      resolvedCode = bJson.code || chosen;
      bundle = bJson.messages || {};
      cache[resolvedCode] = bundle;
    } else {
      const fbRes = await fetch(`/api/v1/locales/${encodeURIComponent(chosen)}`, { cache: 'no-cache' });
      if (fbRes.ok) {
        const bJson = await fbRes.json();
        resolvedCode = bJson.code || chosen;
        bundle = bJson.messages || {};
        cache[resolvedCode] = bundle;
      }
    }
  } catch (_) {}

  if (!locales.length && bundle._meta) {
    locales = [bundle._meta];
  }
  applyHtmlLang(bundle, resolvedCode);
  return {
    defaultCode,
    currentCode: resolvedCode,
    locales,
    cache,
    bundle,
  };
}

async function mountWildCatApp() {
  const i18nInit = await loadInitialI18nState();

  createApp({
    setup() {
      // ───────── Dynamic JSON-Driven Internationalization (i18n) ─────────
      const availableLocales = ref(i18nInit.locales);
      const currentLang = ref(i18nInit.currentCode);
      const activeMessages = ref(i18nInit.bundle);
      const localeBundleCache = i18nInit.cache;

      function t(key, vars = null) {
        const raw = lookupLocalePath(activeMessages.value, key);
        if (raw === undefined) return key;
        return interpolateTemplate(raw, vars, false);
      }

      function tHtml(key, vars = null) {
        const raw = lookupLocalePath(activeMessages.value, key);
        if (raw === undefined) return escapeHtml(key);
        return interpolateTemplate(raw, vars, true);
      }

      async function setLanguage(nextCode) {
        const target = String(nextCode || i18nInit.defaultCode).trim();
        if (!target) return;
        let resolved = target;
        let bundle = localeBundleCache[target];
        if (!bundle) {
          try {
            const res = await fetch(`/api/v1/locales/${encodeURIComponent(target)}`, { cache: 'no-cache' });
            if (res.ok) {
              const data = await res.json();
              resolved = data.code || target;
              bundle = data.messages || {};
              localeBundleCache[resolved] = bundle;
            }
          } catch (_) {}
        }
        if (bundle) {
          currentLang.value = resolved;
          activeMessages.value = bundle;
          if (typeof localStorage !== 'undefined') {
            localStorage.setItem('wc_lang', resolved);
          }
          applyHtmlLang(bundle, resolved);
          await nextTick();
          if (topNav.value === 'studio' && studioMode.value === 'monitor') {
            drawLossChart();
            drawQualityChart();
          }
          if (topNav.value === 'lab' && labMode.value === 'analyze') {
            drawLabCompareChart();
          }
        }
      }

      // ───────── Notifications ─────────
      const toasts = ref([]);
      let toastId = 1;
      function notify(msg, type = 'info') {
        const id = toastId++;
        toasts.value.push({ id, msg, type });
        setTimeout(() => {
          toasts.value = toasts.value.filter((x) => x.id !== id);
        }, 3600);
      }

      // ───────── Algorithmic Session Identity, Roaming Passcode & Admin Auth State ─────────
      function computeClientFingerprint() {
        try {
          const tz =
            (typeof Intl !== 'undefined' &&
              Intl.DateTimeFormat &&
              Intl.DateTimeFormat().resolvedOptions &&
              Intl.DateTimeFormat().resolvedOptions().timeZone) ||
            '';
          const scr = typeof screen !== 'undefined' ? `${screen.width || 0}x${screen.height || 0}x${screen.colorDepth || 0}` : '';
          const nav =
            typeof navigator !== 'undefined'
              ? `${navigator.hardwareConcurrency || 0}|${navigator.platform || ''}`
              : '';
          const raw = `wc_fp_v1|${scr}|${nav}|${tz}`;
          let h1 = 0x811c9dc5;
          let h2 = 0x01000193;
          for (let i = 0; i < raw.length; i++) {
            const c = raw.charCodeAt(i);
            h1 ^= c;
            h1 = Math.imul(h1, 0x01000193) >>> 0;
            h2 ^= c + i;
            h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
          }
          return `${h1.toString(16).padStart(8, '0')}${h2.toString(16).padStart(8, '0')}_${raw}`.slice(0, 120);
        } catch (_) {
          return 'wc_fp_fallback';
        }
      }

      const clientFingerprint = computeClientFingerprint();
      const user = ref(null);
      const buildLabel = ref('v2.3.0');
      const apiDocsOpen = ref(false); // the server says whether /docs would open in this browser (it is not served to outsiders)
      const roamCodeLocal = ref((typeof localStorage !== 'undefined' ? localStorage.getItem('wc_roam_code') : '') || '');
      const machineName = computed(() => (user.value && user.value.username) || '');
      const apiKey = ref('');
      const roamModal = reactive({
        open: false,
        codeInput: '',
        loading: false,
        errorMsg: '',
      });

      const restrictionsEnabled = computed(() => !!(user.value && user.value.restrictions_enabled));
      const isPrivilegedUser = computed(() => {
        if (!user.value) return false;
        return Boolean(!user.value.restrictions_enabled || user.value.role === 'admin' || user.value.is_admin);
      });
      const canUseLab = computed(() => Boolean(user.value && user.value.perm_lab));
      const canUseApi = computed(() => Boolean(user.value && user.value.perm_api));
      const canUseConverter = computed(() => Boolean(user.value && user.value.perm_converter));
      const userMaxEpochs = computed(() => {
        if (!user.value || isPrivilegedUser.value) return 0;
        const cap = Number(user.value.max_epochs || 0);
        return cap > 0 ? cap : 0;
      });
      const userTaskQuotaReached = computed(() => {
        if (!user.value || isPrivilegedUser.value) return false;
        const maxT = Number(user.value.max_tasks ?? 2);
        const usedT = Number(user.value.tasks_used ?? 0);
        return usedT >= maxT;
      });

      const _loadStoredAdminToken = () => {
        if (typeof sessionStorage === 'undefined') return '';
        const raw = (sessionStorage.getItem('wc_admin_token') || '').trim();
        if (raw && raw.split('.').length !== 3) {
          sessionStorage.removeItem('wc_admin_token');
          return '';
        }
        return raw;
      };
      const adminToken = ref(_loadStoredAdminToken());
      const adminPwdInput = ref('');
      const adminAuthLoading = ref(false);
      const adminAuthError = ref('');
      const pwdForm = reactive({ old_password: '', new_password: '' });

      // ───────── Top Navigation, Studio & Experiment Lab Mode ─────────
      const initialNav =
        typeof window !== 'undefined' && window.location && window.location.pathname === '/admin' ? 'admin' : 'studio';
      const topNav = ref(initialNav); // 'studio' | 'lab' | 'assets' | 'tools' | 'help' | 'admin'
      // The sections of the Tools tab in page order ('api' = the account's API key; an edition may add sections before it)
      // and the one shown.
      const toolsSections = ['api'];
      const toolsSection = ref(toolsSections[0]);
      // the tab itself carries the lock only when all it holds is locked: the API section is its only section
      const toolsTabLocked = computed(() => !!(user.value && !canUseApi.value) && toolsSections.length === 1);
      // 'api' was the tab of the API key until it moved into the Tools tab: whatever still goes there lands in that section
      watch(
        topNav,
        (nav) => {
          if (nav !== 'api') return;
          toolsSection.value = 'api';
          topNav.value = 'tools';
        },
        { flush: 'sync' }
      );
      const workspaceReady = ref(false);
      const studioMode = ref('create'); // 'create' | 'monitor'
      const labMode = ref('plan'); // 'plan' | 'analyze'
      const initCategory = ref('scratch'); // 'scratch' | 'checkpoint'

      const queue = reactive({
        gpus: [],
        running: [],
        queued: [],
        queued_count: 0,
        running_count: 0,
        total_gpu_count: 1,
        enabled_gpu_count: 1,
        all_gpus_disabled: false,
        primary_gpu: null,
        secondary_gpu: null,
        idle_gpu_count: 1,
        max_jobs_per_gpu: 1,
        next_auto_wait_sec: 0,
      });
      const tasks = ref([]);
      const labGroupMeta = reactive({ groups: [], order: [] });
      const labDbSummary = reactive({
        tasks: 0,
        groups: 0,
        evals: 0,
        curve_points: 0,
        eval_frame_points: 0,
      });
      function applyServerLabGroups(lg) {
        if (!lg || typeof lg !== 'object') return;
        if (Array.isArray(lg.groups)) labGroupMeta.groups = lg.groups;
        if (Array.isArray(lg.order)) labGroupMeta.order = lg.order;
      }
      const assets = ref([]);
      const selectedDatasetId = ref(null);
      const rawCheckpoints = ref([]);
      const checkpoints = computed(() => {
        if (rawCheckpoints.value.length) return rawCheckpoints.value;
        return [
          {
            path: '',
            label: t('wizard.noCheckpointsSaved'),
            model_size: 'Medium',
            in_channels: 3,
            out_channels: 3,
          },
        ];
      });
      const selectedTaskId = ref(null);
      const extendTargetEpochs = ref(20000);
      const extendInputFocused = ref(false);
      // ───────── Help center: parameter reference · scene presets · strategy evidence (advanced) · hardware speed ─────────
      const HELP_TAB_ORDER = ['reference', 'presets', 'evidence', 'benchmarks'];
      const helpTab = ref('reference');
      const helpMainRef = ref(null);
      const helpDocQuery = ref('');
      const helpEvQuery = ref('');
      const helpFocusId = ref(''); // 'doc-<id>' / 'ev-<id>' while the entry opened from a "?" or a link is highlighted
      const helpReturn = ref(null); // where the "?" / link was clicked: { nav, studioMode, labMode, scrollTop }
      let helpFocusTimer = 0;
      const helpBenchRaw = ref(null);
      const helpBenchError = ref(false);
      const helpStaticLoading = ref(false);
      const gpuBenchDb = ref(null);

      const helpTabs = computed(() => HELP_TAB_ORDER.map((id) => ({ id, lbl: t(`help.tab_${id}`) })));

      function switchHelpTab(tabId) {
        if (HELP_TAB_ORDER.includes(tabId)) {
          helpTab.value = tabId;
          if (helpMainRef.value) helpMainRef.value.scrollTop = 0;
        }
      }

      const helpPrevTab = computed(() => {
        const idx = HELP_TAB_ORDER.indexOf(helpTab.value);
        return idx > 0 ? HELP_TAB_ORDER[idx - 1] : null;
      });

      const helpNextTab = computed(() => {
        const idx = HELP_TAB_ORDER.indexOf(helpTab.value);
        return idx >= 0 && idx < HELP_TAB_ORDER.length - 1 ? HELP_TAB_ORDER[idx + 1] : null;
      });

      async function fetchGpuBenchmarks() {
        try {
          const r = await fetch('/api/v1/gpu-benchmarks', { cache: 'no-cache' });
          if (r.ok) {
            const d = await r.json();
            if (d && d.status === 'ok') gpuBenchDb.value = d;
          }
        } catch (_) {}
      }

      // entries of the active language (main locale + private overlay, merged by the server)
      function docEntry(id) {
        const e = lookupLocaleNode(activeMessages.value, `doc.${id}`);
        return e && typeof e === 'object' ? e : null;
      }
      function evEntry(id) {
        const e = lookupLocaleNode(activeMessages.value, `evidence.${id}`);
        return e && typeof e === 'object' ? e : null;
      }
      const hasDoc = (id) => !!docEntry(id);
      const hasEv = (id) => !!evEntry(id);
      const docTitle = (id) => (docEntry(id) || {}).title || id;
      const docSummary = (id) => (docEntry(id) || {}).summary || '';
      const evTitle = (id) => (evEntry(id) || {}).title || id;
      const evSummary = (id) => (evEntry(id) || {}).summary || '';
      const resolveDocLink = (kind, id) => ((kind === 'doc' ? docEntry(id) : evEntry(id)) || {}).title || '';

      // a parameter entry ends with the link to its test evidence when there is one (same id)
      function docHtml(id) {
        const e = docEntry(id);
        if (!e) return '';
        let html = renderDocMarkup(e.body, resolveDocLink);
        if (evEntry(id)) {
          html += `<p class="wc-doc-more"><a href="#ev-${id}" class="wc-doc-link" data-ev="${id}">${escapeHtml(t('help.evidenceLink'))}</a></p>`;
        }
        return html;
      }
      function evHtml(id) {
        const e = evEntry(id);
        if (!e) return '';
        let html = renderDocMarkup(e.body, resolveDocLink);
        if (docEntry(id)) {
          html += `<p class="wc-doc-more"><a href="#doc-${id}" class="wc-doc-link" data-doc="${id}">${escapeHtml(t('help.docLink', { title: docTitle(id) }))}</a></p>`;
        }
        return html;
      }

      const helpMatches = (query, ...texts) => {
        const q = String(query || '').trim().toLowerCase();
        return !q || texts.some((x) => String(x || '').toLowerCase().includes(q));
      };
      const helpDocGroupsView = computed(() =>
        DOC_GROUPS.map((g) => ({
          id: g.id,
          ids: g.ids.filter((id) => hasDoc(id) && helpMatches(helpDocQuery.value, docTitle(id), docSummary(id), id)),
        })).filter((g) => g.ids.length)
      );
      const helpDocTotal = computed(() => DOC_GROUPS.reduce((n, g) => n + g.ids.filter((id) => hasDoc(id)).length, 0));
      const helpDocShown = computed(() => helpDocGroupsView.value.reduce((n, g) => n + g.ids.length, 0));
      const helpEvGroupsView = computed(() =>
        EVIDENCE_GROUPS.map((g) => ({
          id: g.id,
          ids: g.ids.filter((id) => hasEv(id) && helpMatches(helpEvQuery.value, evTitle(id), evSummary(id), id)),
        })).filter((g) => g.ids.length)
      );
      const helpEvTotal = computed(() => EVIDENCE_GROUPS.reduce((n, g) => n + g.ids.filter((id) => hasEv(id)).length, 0));
      const helpEvShown = computed(() => helpEvGroupsView.value.reduce((n, g) => n + g.ids.length, 0));

      // what a scene preset sets (PRESET_SPECS), labelled with the titles of the parameter reference
      function presetValueRows(pid) {
        const on = t('help.valOn');
        const keep = t('help.valUnchanged');
        if (pid === 'nuke') {
          return [
            [docTitle('model_size'), keep],
            [docTitle('crop_size'), '256'],
            [docTitle('multi_resolution'), on],
            [docTitle('adv_overview'), t('help.valAllOff')],
            [docTitle('epochs'), keep],
          ];
        }
        const s = PRESET_SPECS[pid];
        if (!s) return [];
        const rows = [
          [docTitle('model_size'), s.model_size],
          [docTitle('channels'), t(`wizard.channelPreset${s.channels.replace('->', '')}`)],
          [docTitle('crop_size'), String(s.crop)],
          [docTitle('multi_resolution'), on],
          [docTitle('adv_crop_grid'), `${s.crop_grid} px`],
        ];
        if (s.loss) {
          if (s.loss.charbonnier_ratio) rows.push([docTitle('adv_loss_charbonnier'), String(s.loss.charbonnier_ratio)]);
          if (s.loss.grad_weight) rows.push([docTitle('adv_loss_grad'), String(s.loss.grad_weight)]);
          if (s.loss.hdr_alpha) rows.push([docTitle('adv_loss_hdr'), String(s.loss.hdr_alpha)]);
        }
        if (s.stages) {
          rows.push([docTitle('adv_stages_ratio'), `${s.stages.r_quarter} / ${s.stages.r_half} / ${s.stages.r_full} %`]);
          rows.push([docTitle('adv_stages_warmup'), t('help.valSteps', { n: s.stages.warmup_steps })]);
        }
        if (s.ema) rows.push([docTitle('adv_ema'), `β = ${s.ema.ema_decay}`]);
        if (s.kf_common) {
          rows.push([docTitle('adv_kf_common'), t('help.valKfCommon', { threshold: s.kf_common.threshold, span: s.kf_common.span })]);
        }
        rows.push([t('help.valOthers'), t('help.valOff')]);
        rows.push([docTitle('epochs'), keep]);
        return rows;
      }

      const helpBench = computed(() => {
        if (!helpBenchRaw.value && !helpBenchError.value) return null;
        if (helpBenchError.value || !helpBenchRaw.value) {
          return {
            gpus: [],
            rows: [],
            updated: '',
            setup: t('help.benchLoadFailed'),
            notes: [],
          };
        }
        return {
          gpus: helpBenchRaw.value.gpus || [],
          rows: helpBenchRaw.value.rows || [],
          updated: helpBenchRaw.value.date ? t('help.benchUpdated', { date: helpBenchRaw.value.date }) : '',
          setup: t('help.benchSetup'),
          notes: [t('help.benchNote1'), t('help.benchNote2'), t('help.benchNote3')],
        };
      });

      async function loadHelpStaticData() {
        if (helpStaticLoading.value) return;
        helpStaticLoading.value = true;
        const tasksArr = [];
        if (!gpuBenchDb.value) {
          tasksArr.push(fetchGpuBenchmarks());
        }
        if (!helpBenchRaw.value) {
          helpBenchError.value = false;
          tasksArr.push(
            fetch('/static/benchmarks.json?v=20261114_audit1')
              .then((r) => {
                if (!r.ok) throw new Error('status ' + r.status);
                return r.json();
              })
              .then((d) => {
                helpBenchRaw.value = d;
              })
              .catch(() => {
                helpBenchError.value = true;
              })
          );
        }
        await Promise.all(tasksArr);
        helpStaticLoading.value = false;
      }

      async function openHelp(tabId) {
        if (typeof tabId === 'string' && HELP_TAB_ORDER.includes(tabId)) {
          helpTab.value = tabId;
        }
        topNav.value = 'help';
        await loadHelpStaticData();
        if (helpMainRef.value) helpMainRef.value.scrollTop = 0;
      }

      // a "?", a link inside a rendered entry or a card opens the help center at the entry; Esc / the back button returns
      async function openHelpEntry(kind, id, srcEl) {
        if (topNav.value !== 'help') {
          const main = (srcEl && srcEl.closest && srcEl.closest('main')) || document.querySelector('main');
          helpReturn.value = { nav: topNav.value, studioMode: studioMode.value, labMode: labMode.value, scrollTop: main ? main.scrollTop : 0 };
        }
        if (kind === 'doc') {
          helpTab.value = 'reference';
          helpDocQuery.value = '';
        } else {
          helpTab.value = 'evidence';
          helpEvQuery.value = '';
        }
        topNav.value = 'help';
        loadHelpStaticData();
        await nextTick();
        const el = document.getElementById(`${kind}-${id}`);
        if (el) el.scrollIntoView({ block: 'start' });
        helpFocusId.value = '';
        await nextTick();
        helpFocusId.value = `${kind}-${id}`;
        clearTimeout(helpFocusTimer);
        helpFocusTimer = setTimeout(() => {
          helpFocusId.value = '';
        }, 3200);
      }
      const openHelpDoc = (id, srcEl) => openHelpEntry('doc', id, srcEl);
      const openHelpEvidence = (id, srcEl) => openHelpEntry('ev', id, srcEl);

      async function helpGoBack() {
        const r = helpReturn.value;
        if (!r) return;
        helpReturn.value = null;
        topNav.value = r.nav;
        studioMode.value = r.studioMode;
        labMode.value = r.labMode;
        await nextTick();
        const restore = () => {
          const main = document.querySelector('main');
          if (main) main.scrollTop = r.scrollTop;
        };
        restore();
        setTimeout(restore, 150);
      }

      // links in rendered entries (data-doc / data-ev) and Esc while the help center was opened from the page
      function onHelpLinkClick(ev) {
        const a = ev.target && ev.target.closest ? ev.target.closest('a[data-doc], a[data-ev]') : null;
        if (!a) return;
        ev.preventDefault();
        if (a.dataset.doc) openHelpDoc(a.dataset.doc, a);
        else openHelpEvidence(a.dataset.ev, a);
      }
      function onHelpKeydown(ev) {
        if (ev.key !== 'Escape' || topNav.value !== 'help' || !helpReturn.value) return;
        const tgt = ev.target;
        if (tgt && tgt.tagName === 'INPUT' && tgt.value) return; // the first Esc in a search box only clears it (see the input)
        ev.preventDefault();
        helpGoBack();
      }
      provide('wcHelp', { t, docTitle, docSummary, hasDoc, openDoc: openHelpDoc });
      onMounted(() => {
        document.addEventListener('click', onHelpLinkClick);
        document.addEventListener('keydown', onHelpKeydown);
      });
      watch(topNav, (nav) => {
        if (nav !== 'help') helpReturn.value = null;
      });

      function benchTotalSpeedup(r) {
        const sumInv = (a) => a.reduce((s, v) => s + 1 / v, 0);
        return (sumInv(r.nuke) / sumInv(r.wildcat)).toFixed(1);
      }

      // ───────── Step 1-3 Create Task Form (CopyCat Baseline Defaults) ─────────
      const form = reactive({
        run_name: 'WildCat1',
        preferred_gpu: 'auto',
        source: 'dataset/source',
        target: 'dataset/target',
        model_size: 'Medium',
        in_channels: 3,
        out_channels: 3,
        epochs: 10000,
        batch_size_mode: 'Auto',
        batch_size: 8,
        crop_size: 256,
        multi_resolution: true,
        learning_rate: 0.0001,
        precision: 'adaptive',
        seed: 0,
        working_colorspace: 'ACEScg',
        output_format: 'exr_half',
        initial_weights_mode: 'None',
        checkpoint_path: '',
        checkpoint_interval: 1000,
        preview_interval: 100,
        log_interval: 10,
        // Lossless engine optimisations (identical training semantics to CopyCat)
        opt_gpu_resident: true,
        opt_cuda_graph: true,
        opt_fused_adam: true,
        opt_channels_last: true,
        opt_cudnn_benchmark: true,
        opt_async_io: true,
        preview_gpu: 'auto',
      });

      const channelPreset = ref('3->3');

      // ───────── Advanced Hyperparameters Lab (Default Collapsed & All Off = 100% Nuke Native) ─────────
      const advPanelOpen = ref(false);
      const activeAdvPreset = ref('nuke');
      const advItemOpen = reactive({
        crop_grid: false,
        loss_family: false,
        stage_schedule: false,
        ema_lr: false,
        augment: false,
        kf_common: false,
      });
      const advRawTexOpen = reactive({});
      const adv = reactive(createDefaultAdvConfig());
      const advMathHtml = ADV_MATH_HTML;
      const advLatexSrc = ADV_LATEX;

      const ADV_KEYS = ['crop_grid', 'loss_family', 'stage_schedule', 'ema_lr', 'augment', 'kf_common'];

      // averaged weights (module 04): read-out of the decay as "about the last N steps" and the half-life
      const advEmaPreview = computed(() => {
        const d = Math.max(0.9, Math.min(0.99999, Number(adv.ema_lr.ema_decay || 0.9995)));
        const effSteps = Math.round(1 / Math.max(1e-6, 1 - d));
        const halfLife = Math.round(Math.LN2 / Math.max(1e-6, -Math.log(d)));
        return { decay: d, effSteps, halfLife };
      });

      // The 2-px grid is no longer offered: in the 2026-11 re-validation it was never better than 1 px (0.92 dB worse on the
      // clean-up shot, no difference on keying / defocus / degrain) at the same speed. Its button only appears when a loaded
      // configuration already uses it, so that such a job still shows what it runs with. The backend accepts 2 as before.
      const advCropGridOptions = computed(() => {
        const opts = [
          { v: 1, title: t('adv.cg.opt1Title'), desc: t('adv.cg.opt1Desc') },
          { v: 4, title: t('adv.cg.opt4Title'), desc: t('adv.cg.opt4Desc') },
        ];
        if (Number(adv.crop_grid.full_res_grid) === 2) {
          opts.splice(1, 0, { v: 2, title: t('adv.cg.opt2Title'), desc: t('adv.cg.opt2Desc') });
        }
        return opts;
      });

      function setCropGridAnchor(a) {
        adv.crop_grid.anchor = a === 'top' ? 'top' : 'bottom';
        if (Number(adv.crop_grid.full_res_grid) === 4) adv.crop_grid.enabled = adv.crop_grid.anchor === 'top';
      }

      function isAdvModuleEffective(k) {
        if (!adv[k] || !adv[k].enabled) return false;
        if (k === 'crop_grid') {
          return Number(adv.crop_grid.full_res_grid || 4) !== 4 || adv.crop_grid.anchor === 'top';
        }
        if (k === 'stage_schedule') {
          return !!form.multi_resolution;
        }
        if (k === 'loss_family') {
          const c = Math.max(0, Number(adv.loss_family.charbonnier_ratio) || 0);
          const g = Math.max(0, Number(adv.loss_family.grad_weight) || 0);
          const h = Number(form.out_channels) === 1 ? 0 : Math.max(0, Number(adv.loss_family.hdr_alpha) || 0);
          return c > 0 || g > 0 || h > 0;
        }
        if (k === 'augment') {
          const fl = Math.max(0, Number(adv.augment.hflip_prob) || 0);
          const ev =
            Number(form.in_channels) < 3 && Number(form.out_channels) < 3
              ? 0
              : Math.max(0, Number(adv.augment.exposure_ev) || 0);
          return fl > 0 || ev > 0;
        }
        if (k === 'kf_common') {
          // RGB -> RGB only, the same rule as the trainer. A missing plate or too few keyframes do not switch it off here:
          // the trainer then trains as usual and reports why (task page); the card says so beforehand (kfCommonBlockReason)
          return Number(form.in_channels) === 3 && Number(form.out_channels) === 3;
        }
        return true;
      }

      const advActiveCount = computed(() => {
        return ADV_KEYS.reduce((n, k) => n + (isAdvModuleEffective(k) ? 1 : 0), 0);
      });

      // Keyframe commonality: why it would not take effect for the current model / dataset ('' = nothing known against it)
      const kfCommonBlockReason = computed(() => kfCommonBlocker(form.in_channels, form.out_channels, selectedDataset.value));
      const kfCommonLimits = KF_COMMON_LIMITS;

      // RAFT as the optical flow of keyframe commonality is offered only where the server can run it right now. That is
      // the capability state of the Tools tab (kfCaps, filled by loadKfCaps from GET /api/v1/tools/kf-caps): the form
      // reads the same state and asks for it at most once, the first time the card is opened - and not at all when the
      // Tools tab has already found RAFT available. (typeof: the Tools tab is not part of every edition.)
      const kfCommonRaftOffered = computed(() => typeof kfCaps !== 'undefined' && !!kfCaps.raft);
      let kfFlowCapsAsked = false;
      let kfFlowCapsKnown = false;
      function ensureKfFlowCaps() {
        if (kfFlowCapsAsked || typeof kfCaps === 'undefined' || typeof loadKfCaps !== 'function') return;
        kfFlowCapsAsked = true;
        if (kfCaps.raft) {
          kfFlowCapsKnown = true;
          return;
        }
        Promise.resolve(loadKfCaps()).then(() => {
          kfFlowCapsKnown = true;
          syncKfCommonParams();
        });
      }

      function syncKfCommonParams() {
        Object.assign(adv.kf_common, clampKfCommon(adv.kf_common));
        // RAFT chosen (for example in a restored config) where the server is known not to be able to run it: the default.
        // While that is not known the choice is kept - the trainer then uses DIS on its own and the task page says so.
        if (kfFlowCapsKnown && !kfCommonRaftOffered.value) adv.kf_common.flow = 'dis';
      }

      function syncLossFamilyEnabled() {
        if (Number(form.out_channels) === 1) {
          adv.loss_family.hdr_alpha = 0;
        }
        const c = Math.max(0, Number(adv.loss_family.charbonnier_ratio) || 0);
        const g = Math.max(0, Number(adv.loss_family.grad_weight) || 0);
        const h = Number(form.out_channels) === 1 ? 0 : Math.max(0, Number(adv.loss_family.hdr_alpha) || 0);
        adv.loss_family.enabled = c > 0 || g > 0 || h > 0;
      }

      function onLossFamilyToggle() {
        const oc = Number(form.out_channels);
        if (oc === 1) {
          adv.loss_family.hdr_alpha = 0;
        }
        if (adv.loss_family.enabled) {
          const c = Math.max(0, Number(adv.loss_family.charbonnier_ratio) || 0);
          const g = Math.max(0, Number(adv.loss_family.grad_weight) || 0);
          const h = oc === 1 ? 0 : Math.max(0, Number(adv.loss_family.hdr_alpha) || 0);
          if (c <= 0 && g <= 0 && h <= 0) {
            // re-validated starting values: gradient term only (0.15 measured on keying, 0.05 elsewhere)
            adv.loss_family.charbonnier_ratio = 0;
            adv.loss_family.grad_weight = oc === 4 ? 0.15 : 0.05;
            adv.loss_family.hdr_alpha = 0;
          }
        }
      }

      function syncAugmentEnabled() {
        if (Number(form.in_channels) < 3 && Number(form.out_channels) < 3) {
          adv.augment.exposure_ev = 0;
        }
        const fl = Math.max(0, Number(adv.augment.hflip_prob) || 0);
        const ev =
          Number(form.in_channels) < 3 && Number(form.out_channels) < 3
            ? 0
            : Math.max(0, Number(adv.augment.exposure_ev) || 0);
        adv.augment.enabled = fl > 0 || ev > 0;
      }

      function onAugmentToggle() {
        const lockEv = Number(form.in_channels) < 3 && Number(form.out_channels) < 3;
        if (lockEv) {
          adv.augment.exposure_ev = 0;
        }
        if (adv.augment.enabled) {
          const fl = Math.max(0, Number(adv.augment.hflip_prob) || 0);
          const ev = lockEv ? 0 : Math.max(0, Number(adv.augment.exposure_ev) || 0);
          if (fl <= 0 && ev <= 0) {
            adv.augment.hflip_prob = 0.5;
            adv.augment.exposure_ev = 0.0;
          }
        }
      }

      const advAnyItemExpanded = computed(() => {
        return ADV_KEYS.some((k) => !!advItemOpen[k]);
      });

      function toggleAdvItem(key) {
        advItemOpen[key] = !advItemOpen[key];
        if (key === 'kf_common' && advItemOpen[key]) ensureKfFlowCaps();
      }

      function toggleAdvAllItems(openState) {
        const target = typeof openState === 'boolean' ? openState : !advAnyItemExpanded.value;
        for (const k of ADV_KEYS) {
          advItemOpen[k] = target;
        }
        if (target) ensureKfFlowCaps();
      }

      function toggleRawLatex(texKey) {
        advRawTexOpen[texKey] = !advRawTexOpen[texKey];
      }

      async function copyLatexSource(texKey) {
        const raw = ADV_LATEX[texKey] || '';
        if (!raw) return;
        try {
          await navigator.clipboard.writeText(raw);
          notify(t('msg.copiedLatex'), 'success');
        } catch (e) {
          notify(t('msg.copyFailedManual'), 'warning');
        }
      }

      const cropGridPhaseHint = computed(() => {
        const h = Number(selectedDataset.value && selectedDataset.value.height);
        return h > 0 && h % 4 !== 0 && !isAdvModuleEffective('crop_grid');
      });

      function resetAdvToNukeNative(silent = false) {
        activeAdvPreset.value = 'nuke';
        const fresh = createDefaultAdvConfig();
        for (const k of ADV_KEYS) {
          Object.assign(adv[k], fresh[k]);
        }
        if (Number(form.out_channels) === 1) {
          adv.loss_family.hdr_alpha = 0;
        }
        if (Number(form.in_channels) < 3 && Number(form.out_channels) < 3) {
          adv.augment.exposure_ev = 0;
        }
        if (!silent) {
          notify(t('msg.resetAdvDone'), 'info');
        }
      }

      function applyEmpiricalPresetCore(presetId) {
        const setPresetModelAndCh = (modelSz, chPreset) => {
          if (!isModelSizeLocked.value && modelSz) {
            form.model_size = modelSz;
          }
          if (!isChannelsLocked.value && chPreset && isChannelPresetAllowed(chPreset)) {
            const [ic, oc] = chPreset.split('->').map(Number);
            form.in_channels = ic;
            form.out_channels = oc;
            channelPreset.value = chPreset;
          }
        };

        resetAdvToNukeNative(true);
        if (presetId === 'nuke') {
          form.crop_size = 256;
          form.multi_resolution = true;
          activeAdvPreset.value = 'nuke';
          return true;
        }

        form.multi_resolution = true;
        const dsMinSide =
          selectedDataset.value && Number(selectedDataset.value.width) >= 32 && Number(selectedDataset.value.height) >= 32
            ? Math.min(Number(selectedDataset.value.width), Number(selectedDataset.value.height))
            : 1080;
        const safeCrop = (want) => {
          const maxLeg = Math.max(32, Math.min(512, Math.floor(dsMinSide / 16) * 16));
          if (want >= 384 && dsMinSide < 720) return Math.min(maxLeg, 256);
          return Math.max(32, Math.min(maxLeg, Math.floor(want / 16) * 16));
        };

        // The values of every preset are PRESET_SPECS (module scope); the help center lists the same table.
        const spec = PRESET_SPECS[presetId];
        if (!spec) {
          return false;
        }
        adv.crop_grid.enabled = true;
        adv.crop_grid.full_res_grid = spec.crop_grid;
        adv.loss_family.enabled = false;
        adv.stage_schedule.enabled = false;
        adv.augment.enabled = false;
        adv.ema_lr.enabled = false;
        adv.kf_common.enabled = false;
        setPresetModelAndCh(spec.model_size, spec.channels);
        form.crop_size = safeCrop(spec.crop);
        if (spec.loss) {
          adv.loss_family.enabled = true;
          adv.loss_family.charbonnier_ratio = spec.loss.charbonnier_ratio;
          adv.loss_family.grad_weight = spec.loss.grad_weight;
          adv.loss_family.hdr_alpha = Number(form.out_channels) === 1 ? 0 : spec.loss.hdr_alpha;
        }
        if (spec.stages) {
          adv.stage_schedule.enabled = true;
          Object.assign(adv.stage_schedule, spec.stages);
        }
        if (spec.ema) Object.assign(adv.ema_lr, spec.ema);
        if (spec.kf_common) Object.assign(adv.kf_common, spec.kf_common, { enabled: true });
        activeAdvPreset.value = presetId;
        return true;
      }

      function applyAdvPreset(presetId) {
        if (!applyEmpiricalPresetCore(presetId)) return;
        if (presetId === 'nuke') {
          notify(t('msg.presetNukeLoaded'), 'info');
          return;
        }
        advPanelOpen.value = true;
        // the face-edit preset is not part of the 2026-11 re-validation, so it does not get the toast that says so
        const loadedKey = presetId === 'face_edit' ? 'msg.presetFaceLoaded' : 'msg.presetEmpiricalLoaded';
        notify(t(loadedKey, { name: t(`adv.preset_${presetId}`) }), 'success');
      }

      // "Load this preset" in the help center: a fresh training form with the preset applied
      function applyHelpPreset(pid) {
        if (!PRESET_IDS.includes(pid)) return;
        initCategory.value = 'scratch';
        form.initial_weights_mode = 'None';
        form.checkpoint_path = '';
        applyEmpiricalPresetCore(pid);
        advPanelOpen.value = true;
        studioMode.value = 'create';
        topNav.value = 'studio';
        const loadedKey = pid === 'face_edit' ? 'msg.helpRecipeFaceLoadedToast' : 'msg.helpRecipeLoadedToast';
        notify(t(loadedKey, { name: docTitle(`preset_${pid}`) }), 'success');
      }

      function onStageRatioInput(changedKey) {
        if (!form.multi_resolution) {
          adv.stage_schedule.enabled = false;
          return;
        }
        adv.stage_schedule.enabled = true;
        let q = Math.max(0, Math.min(90, Math.round(Number(adv.stage_schedule.r_quarter) || 0)));
        let h = Math.max(0, Math.min(90, Math.round(Number(adv.stage_schedule.r_half) || 0)));
        let f = Math.max(5, Math.min(100, Math.round(Number(adv.stage_schedule.r_full) || 0)));
        if (changedKey === 'r_quarter') {
          if (q + h > 95) h = Math.max(0, 95 - q);
          f = Math.max(5, 100 - q - h);
        } else if (changedKey === 'r_half') {
          if (q + h > 95) q = Math.max(0, 95 - h);
          f = Math.max(5, 100 - q - h);
        } else {
          const rem = 100 - f;
          const prevSum = q + h;
          if (prevSum > 0) {
            q = Math.round((q / prevSum) * rem);
            h = Math.max(0, rem - q);
          } else {
            q = Math.floor(rem * 0.4);
            h = rem - q;
          }
        }
        adv.stage_schedule.r_quarter = q;
        adv.stage_schedule.r_half = h;
        adv.stage_schedule.r_full = 100 - q - h;
      }

      const advStageStepsPreview = computed(() => {
        const T = Math.max(1, Number(computedTotalSteps.value || 1000));
        const nukeS = Math.ceil(T / 3);
        const nuke1 = Math.min(T, nukeS);
        const nuke2 = Math.max(0, Math.min(T, 2 * nukeS) - nuke1);
        const nuke3 = Math.max(0, T - nuke1 - nuke2);
        const rq = Math.max(0, Number(adv.stage_schedule.r_quarter || 0));
        const rh = Math.max(0, Number(adv.stage_schedule.r_half || 0));
        const rf = Math.max(0, Number(adv.stage_schedule.r_full || 0));
        const denom = Math.max(1e-6, rq + rh + rf);
        const e1 = Math.min(T, Math.max(0, Math.round((T * rq) / denom)));
        const e2 = Math.min(T, Math.max(e1, Math.round((T * (rq + rh)) / denom)));
        const s1 = e1;
        const s2 = Math.max(0, e2 - e1);
        const s3 = Math.max(0, T - e2);
        return { T, nuke1, nuke2, nuke3, s1, s2, s3 };
      });

      const advHdrPreview = computed(() => {
        const a = Math.max(0, Math.min(2.0, Number(adv.loss_family.hdr_alpha || 0)));
        const wShadow = Math.pow(1 / (1 + 0.1), 2 * a);
        const wMid = Math.pow(1 / (1 + 1.0), 2 * a);
        const wHigh = Math.pow(1 / (1 + 12.0), 2 * a);
        return {
          alpha: a,
          shadowPct: (wShadow * 100).toFixed(0),
          midPct: (wMid * 100).toFixed(0),
          highPct: (wHigh * 100).toFixed(2),
        };
      });

      const advExposurePreview = computed(() => {
        const ev = Math.max(0, Math.min(2, Number(adv.augment.exposure_ev || 0)));
        return {
          low: Math.pow(2, -ev).toFixed(2),
          high: Math.pow(2, ev).toFixed(2),
        };
      });

      // ───────── Monitor, 3x3 Contact Sheet & Model Versions State ─────────
      const telemetry = reactive({
        steps: [],
        vals: [],
        tail: [],
        latest: {},
      });
      const viewerModes = computed(() => [
        { id: 'preview', label: t('monitor.modePreview') },
        { id: 'source', label: t('monitor.modeSource') },
        { id: 'target', label: t('monitor.modeTarget') },
      ]);
      const viewer = reactive({
        mode: 'preview',
        pairs: [],
        selectedPair: null,
        frames: [],
        index: 0,
        playing: false,
        fps: 12,
        zoom: 1,
        panX: 0,
        panY: 0,
        dragging: false,
        dragStartX: 0,
        dragStartY: 0,
      });
      const contactViewer = reactive({
        frames: [],
        index: 0,
        playing: false,
        fps: 8,
        zoom: 1,
        panX: 0,
        panY: 0,
        dragging: false,
        dragStartX: 0,
        dragStartY: 0,
        showOverlay: true,
        pixelated: true,
        cropSize: 256,
        nukePrefix: '',
      });
      const taskModels = reactive({
        nuke_prefix: '',
        zip_filename: '',
        count: 0,
        items: [],
        has_eval_sequence: false,
      });
      // What the optional keyframe-commonality step did for the task on screen (telemetry field kf_common; null = the run
      // folder has no kf_common.json, i.e. the module was not asked for)
      const taskKfCommon = ref(null);
      // What was measured on the in-between frames while the task on screen trained (telemetry field quality; null = the
      // run has no such report and the card is not shown)
      const taskQuality = ref(null);
      const qualityCanvas = ref(null);
      // hoverPoint / hoverFrame: index of the measurement / of the per-frame bar under the pointer or the arrow keys (-1 = none)
      const qualityUi = reactive({ helpOpen: false, tableOpen: false, hoverPoint: -1, hoverLeft: 0, hoverFrame: -1 });
      const chartCfg = reactive({ logScale: true, smooth: 0.6 });
      const lossCanvas = ref(null);
      const logPreRef = ref(null);
      const logAutoScroll = ref(true);

      // ───────── Dataset / Validation Sequence Upload Modal & Scrubber State ─────────
      const uploadModal = reactive({
        open: false,
        mode: 'create', // 'create' | 'attach_seq'
        targetAsset: null,
        kind: 'paired_dataset',
        name: '',
        taskType: '',
        inputColorspace: 'auto',
        enableCheck: true,
        strategy: 'standard',
        maxErrorPct: 35,
        sourceFiles: [],
        targetFiles: [],
        seqFiles: [],
        errorMsg: '',
        uploading: false,
      });

      const datasetPickerModal = reactive({
        open: false,
        mode: 'train', // 'train' | 'lab_eval'
        query: '',
        seqFilter: 'all', // 'all' | 'with_seq'
        taskTypeFilter: 'all', // 'all' | task_type
      });

      const assetFramesCache = reactive({});

      // ───────── Tools & Admin State ─────────
      const adminSettings = reactive({
        max_jobs_per_gpu: 1,
        savingGpuPool: false,
        gpu_pool: {
          gpus: [],
          total_gpu_count: 0,
          enabled_gpu_count: 0,
          all_gpus_disabled: false,
          primary_gpu: null,
          secondary_gpu: null,
          disabled_gpus: [],
        },
        enable_temp_user_restrictions: true,
        default_user_max_tasks: 2,
        default_user_max_epochs: 500000,
        max_running_per_user: 1,
        togglingRestrictions: false,
        savingQuotas: false,
        data_cache_root: '',
        migrate_existing: false,
        savingStorage: false,
        storage: {
          data_cache_root: '',
          is_custom: false,
          default_root_hint: '',
          effective_paths: {
            users_dir: '',
            run_dir: '',
            config_dir: '',
            cache_dir: '',
            db_path: '',
          },
          disk: {
            total_gb: 0,
            used_gb: 0,
            free_gb: 0,
            users_mb: 0,
            run_mb: 0,
            cache_mb: 0,
            db_mb: 0,
          },
        },
        db_summary: {
          tasks: 0,
          groups: 0,
          evals: 0,
          curve_points: 0,
          eval_frame_points: 0,
        },
      });
      const adminUsers = ref([]);
      const adminTasks = ref([]);
      const newUser = reactive({
        username: '',
        note: '',
        perm_lab: false,
        perm_api: false,
      });

      // ───────── Computed Properties & Architecture Locking Rules ─────────
      const pairedAssets = computed(() => {
        return assets.value.filter((a) => a.kind !== 'eval_sequence');
      });

      const evalSequenceAssets = computed(() => {
        const unsup = assets.value.filter((a) => a.kind === 'eval_sequence');
        const paired = assets.value.filter((a) => a.kind !== 'eval_sequence');
        return [...unsup, ...paired];
      });

      const selectedDataset = computed(() => {
        return pairedAssets.value.find((a) => a.id === selectedDatasetId.value) || pairedAssets.value[0] || null;
      });

      const hasSelectedDatasetEvalSeq = computed(() => {
        return !!(selectedDataset.value && selectedDataset.value.has_eval_sequence);
      });

      const selectedCkptMeta = computed(() => {
        if (initCategory.value !== 'checkpoint' || !form.checkpoint_path) return null;
        return checkpoints.value.find((c) => c.path === form.checkpoint_path) || null;
      });


      // Model Size is LOCKED when loading a .pt checkpoint (or a fixed-size built-in preset)
      const isModelSizeLocked = computed(() => {
        if (initCategory.value === 'checkpoint') return true;
        return false;
      });

      // Channels (in_channels -> out_channels) are LOCKED whenever any non-scratch weight is selected
      const isChannelsLocked = computed(() => {
        return initCategory.value === 'checkpoint' || initCategory.value === 'pretrained';
      });

      const datasetFrameCount = computed(() => {
        return Math.max(1, Number((selectedDataset.value && selectedDataset.value.frame_count) || 8));
      });

      const effectiveBatchSize = computed(() => {
        const nFrames = datasetFrameCount.value;
        const req = form.batch_size_mode === 'Auto' ? 8 : Math.max(1, Math.floor(Number(form.batch_size) || 8));
        return Math.min(nFrames, req);
      });

      const isBatchClampedByFrames = computed(() => {
        const hasRealCount = Boolean(selectedDataset.value && selectedDataset.value.frame_count);
        if (!hasRealCount) return false;
        const req = form.batch_size_mode === 'Auto' ? 8 : Math.max(1, Math.floor(Number(form.batch_size) || 8));
        return datasetFrameCount.value < req;
      });

      const stageCropPreview = computed(() => {
        const c = Number(form.crop_size) || 256;
        const ds = selectedDataset.value;
        const minSide =
          ds && Number(ds.width) >= 32 && Number(ds.height) >= 32
            ? Math.min(Number(ds.width), Number(ds.height))
            : Infinity;
        const calcStageCrop = (scale) => {
          const cScaled = Math.max(32, Math.floor(Math.floor(c / scale) / 16) * 16);
          if (!Number.isFinite(minSide)) return cScaled;
          const sideScaled = Math.max(32, Math.floor(Math.floor(minSide / scale) / 16) * 16);
          return Math.min(cScaled, sideScaled);
        };
        if (!form.multi_resolution) return `${calcStageCrop(1)}px`;
        return `${calcStageCrop(4)}→${calcStageCrop(2)}→${calcStageCrop(1)}`;
      });

      const computedTotalSteps = computed(() => {
        const nFrames = Math.max(1, Number((selectedDataset.value && selectedDataset.value.frame_count) || 1));
        const reqBs = form.batch_size_mode === 'Auto' ? 8 : Math.max(1, Math.floor(Number(form.batch_size) || 8));
        const bs = Math.min(nFrames, reqBs);
        const ep = Math.max(1, Number(form.epochs) || 1);
        return Math.max(1, Math.ceil((ep * nFrames) / bs));
      });

      const initSummaryText = computed(() => {
        if (initCategory.value === 'scratch') return 'None';
        if (initCategory.value === 'pretrained') return form.initial_weights_mode;
        return form.checkpoint_path || 'Checkpoint';
      });

      const activeTask = computed(() => {
        return tasks.value.find((x) => x.id === selectedTaskId.value) || tasks.value[0] || null;
      });

      // Keyframe commonality block of the task page: only the report that was fetched for the task on screen
      const activeTaskKfCommon = computed(() => {
        const rep = taskKfCommon.value;
        return rep && activeTask.value && rep.task_id === activeTask.value.id ? rep : null;
      });

      // a 0..1 share as "NN%" ('—' when the report has no number); digits 'area' = one decimal, two below 1 %
      // (one keyframe with a small local outlier among several that agree averages out well below 1 %)
      function kfCommonPct(v, digits = 0) {
        const x = Number(v);
        if (v == null || !Number.isFinite(x)) return '—';
        const pct = x * 100;
        if (digits === 'area') return `${pct.toFixed(pct > 0 && pct < 1 ? 2 : 1)}%`;
        return `${pct.toFixed(digits)}%`;
      }

      // Plain words for the reason code of a report that says "not applied" (codes: wildcat.kfprop.KfError)
      function kfCommonReasonText(code) {
        const hit = lookupLocalePath(activeMessages.value, `monitor.kfcReason_${String(code || '')}`);
        return hit !== undefined ? hit : t('monitor.kfcReason_other');
      }

      // Why RAFT could not run, in plain words, for the code a report carries in flow_fallback (wildcat.kfprop.RaftUnavailable)
      function kfCommonFlowWhy(code) {
        const hit = lookupLocalePath(activeMessages.value, `monitor.kfcFlowWhy_${String(code || '')}`);
        return hit !== undefined ? hit : t('monitor.kfcFlowWhy_failed');
      }

      // ───────── Training quality card of the task page (telemetry field "quality") ─────────
      // only the report that was fetched for the task on screen
      const activeTaskQuality = computed(() => {
        const q = taskQuality.value;
        return q && activeTask.value && q.task_id === activeTask.value.id ? q : null;
      });

      // The two frame distances of the drift measures: header.gap1 / header.gap5 of the report (1 and 5 plate frames
      // unless the report says otherwise, e.g. on a shot that is measured on every third frame). Every text of the card
      // that names them gets them from here ({g1}, {g5}) instead of having "1" and "5" written into it.
      const qualityGaps = computed(() => {
        const q = activeTaskQuality.value;
        const header = q && q.header && typeof q.header === 'object' ? q.header : {};
        const gap = (v, fallback) => {
          const x = qualityNum(v);
          return x !== null && x > 0 ? x : fallback;
        };
        return { g1: gap(header.gap1, 1), g5: gap(header.gap5, 5) };
      });

      function qualityT(key, vars = null) {
        return t(key, Object.assign({}, qualityGaps.value, vars || {}));
      }

      // the six measures of the card with their name and the doc entry that explains them (the "?" of the tile);
      // a charted measure carries the key of its curve (colour + marker), "background change" has no curve
      const qualityMeasures = computed(() =>
        ['fit', 'strength', 'detail', 'look', 'tex', 'bg'].map((id) => ({
          id,
          series: QUALITY_SERIES.find((s) => s.id === id) || null,
          label: qualityT('monitor.qTile_' + id),
          doc: QUALITY_DOC[id],
        }))
      );

      // Plain words for the reason code of a report that says "not measured". The codes a dataset shares with keyframe
      // commonality (no plate, a keyframe that is not a plate frame, ...) read the same sentence as there.
      function qualityReasonText(code) {
        const c = String(code || '');
        if (c) {
          for (const key of [`monitor.qReason_${c}`, `monitor.kfcReason_${c}`]) {
            const hit = lookupLocalePath(activeMessages.value, key);
            if (typeof hit === 'string') return hit;
          }
        }
        return t('monitor.qReason_other');
      }

      // Everything the card shows, as plain values and strings:
      //   null                      no report -> no card
      //   { state: 'unavailable' }  one line with the reason
      //   { state: 'waiting' }      measured, but no point yet
      //   { state: 'ready', ... }   latest numbers (tiles), hints, curves, per-frame strip, table rows
      const qualityView = computed(() => {
        const q = activeTaskQuality.value;
        if (!q) return null;
        if (!q.available) {
          const code = String(q.reason || '');
          return {
            state: 'unavailable',
            reasonText: qualityReasonText(code),
            how: code === 'no_plate' ? qualityT('monitor.qNoPlateHow') : '',
            detail: String(q.detail || '').slice(0, 400),
          };
        }
        let points = qualityPoints(q);
        const latestRaw = q.latest && typeof q.latest === 'object' ? q.latest : null;
        const latest = qualityPoint(latestRaw) || (points.length ? points[points.length - 1] : null);
        if (!latest) return { state: 'waiting' };
        if (!points.length) points = [latest];
        const header = q.header && typeof q.header === 'object' ? q.header : {};
        const floor = header.floor && typeof header.floor === 'object' ? header.floor : {};
        const floor1 = qualityNum(floor.tex_drift1);
        const floor5 = qualityNum(floor.tex_drift5);
        const F = qualityFmt;
        const weakFrame = latest.strength_min_frame !== null ? Math.round(latest.strength_min_frame) : null;

        const P = qualityPct;
        // every tile: the value with its unit, one line saying what the scale means, then the details
        // (a detail line is { text, doc }: doc = the entry its "?" opens, '' for none)
        const tile = (id, value, unit, subs) =>
          Object.assign({}, qualityMeasures.value.find((m) => m.id === id), {
            value,
            unit: unit || '',
            scale: qualityT('monitor.qScale_' + id),
            subs: (subs || []).filter(Boolean).map((s) => (typeof s === 'string' ? { text: s, doc: '' } : s)),
          });
        const tiles = [
          tile('fit', F(latest.kf_fit_roi_db, 1), 'dB', [latest.kf_fit_db !== null ? qualityT('monitor.qTileSub_fitWhole', { v: F(latest.kf_fit_db, 1) }) : '']),
          tile('strength', P(latest.strength_mean), qualityT('monitor.qUnitMean'), [
            latest.strength_min === null
              ? ''
              : weakFrame !== null
                ? qualityT('monitor.qTileSub_weakestAt', { v: P(latest.strength_min), frame: weakFrame })
                : qualityT('monitor.qTileSub_weakest', { v: P(latest.strength_min) }),
          ]),
          tile('detail', P(latest.detail), '', [
            latest.texture !== null ? { text: qualityT('monitor.qTileSub_texture', { v: P(latest.texture) }), doc: 'metric_q_texture' } : '',
          ]),
          tile('look', P(latest.look_drift5), qualityT('monitor.qUnitOver5'), []),
          tile('tex', P(latest.tex_drift1), qualityT('monitor.qUnitOver1'), [
            latest.tex_drift5 !== null ? qualityT('monitor.qTileSub_tex5', { v: P(latest.tex_drift5) }) : '',
            floor1 !== null || floor5 !== null ? qualityT('monitor.qTileSub_floor', { v: P(floor1) + (floor5 !== null ? ' / ' + P(floor5) : '') }) : '',
          ]),
          tile('bg', P(latest.bg_change, 2), '', []),
        ];

        // the only two hints: no verdicts, there are no validated thresholds for these measures. The first one points to
        // the keyframe-coverage tool only in an edition that has the Tools tab (typeof: that tab is not part of every edition).
        // ... and only once the measurement comes from the full-resolution stage: before that the detail is naturally
        // low and the strength is still rising, so a hint would only be noise (a run stopped earlier gets none).
        const bounds = taskStageBoundaries();
        const fullResFrom = bounds.length ? bounds[bounds.length - 1].step : 0;
        const hintsDue = !(Number(latest.step) < fullResFrom);
        const hints = [];
        if (hintsDue && latest.strength_min !== null && latest.strength_min < QUALITY_HINT_BELOW) {
          const plain = typeof kfCover === 'undefined' ? 'Plain' : '';
          hints.push(weakFrame !== null ? qualityT('monitor.qHintWeak' + plain, { frame: weakFrame }) : qualityT('monitor.qHintWeakNoFrame' + plain));
        }
        if (hintsDue && latest.detail !== null && latest.detail < QUALITY_HINT_BELOW) hints.push(qualityT('monitor.qHintBlur'));

        const strip = qualityStrip(latestRaw, header);
        const evaluated = strip ? strip.bars.length : Array.isArray(header.frames) ? header.frames.length : 0;
        const stride = qualityNum(header.stride);
        let info = qualityT('monitor.qLatestInfo', { step: latest.step });
        if (evaluated > 0) {
          info +=
            ' · ' +
            (stride !== null && stride > 1
              ? qualityT('monitor.qLatestFramesStride', { frames: evaluated, stride })
              : qualityT('monitor.qLatestFrames', { frames: evaluated }));
        }

        const charted = QUALITY_SERIES.filter((s) => s.plot < 2);
        const clamped = points.some((p) => charted.some((s) => p[s.field] !== null && p[s.field] > QUALITY_AXIS_MAX)) || !!(strip && strip.over);

        const rows = points
          .slice()
          .reverse()
          .map((p) => ({
            step: p.step,
            fit: `${F(p.kf_fit_roi_db, 1)} / ${F(p.kf_fit_db, 1)}`,
            strength: `${P(p.strength_mean)} / ${P(p.strength_min)}` + (p.strength_min_frame !== null ? ` (${Math.round(p.strength_min_frame)})` : ''),
            detail: `${P(p.detail)} / ${P(p.texture)}`,
            look: P(p.look_drift5),
            tex: `${P(p.tex_drift1)} / ${P(p.tex_drift5)}`,
            bg: P(p.bg_change, 2),
            seconds: F(p.seconds, 1),
          }));

        return { state: 'ready', points, latest, tiles, hints, info, strip, floor1, clamped, rows };
      });

      // the measurement under the pointer (or reached with the arrow keys): the box next to the crosshair
      const qualityHoverPoint = computed(() => {
        const v = qualityView.value;
        const p = v && v.state === 'ready' ? v.points[qualityUi.hoverPoint] : null;
        if (!p) return null;
        const onRight = qualityUi.hoverLeft > 55;
        return {
          step: p.step,
          rows: QUALITY_SERIES.map((s) => ({
            id: s.id,
            series: s,
            value: qualitySeriesText(s, p[s.field]),
          })),
          style: onRight ? { right: `${100 - qualityUi.hoverLeft}%`, marginRight: '10px' } : { left: `${qualityUi.hoverLeft}%`, marginLeft: '10px' },
        };
      });

      // the label above the per-frame strip: the frame under the pointer, otherwise the weakest frame
      const qualityStripCallout = computed(() => {
        const v = qualityView.value;
        const strip = v && v.state === 'ready' ? v.strip : null;
        if (!strip) return null;
        const hovered = strip.bars[qualityUi.hoverFrame] || null;
        const b = hovered || strip.weakest;
        if (!b) return null;
        return {
          text: t(b.weakest ? 'monitor.qStripWeakLabel' : 'monitor.qFrameTip', { frame: b.frame, value: b.text }),
          style: b.center < 12 ? { left: '0' } : b.center > 88 ? { right: '0' } : { left: `${b.center}%`, transform: 'translateX(-50%)' },
          center: b.center,
          gap: 100 - b.h,
        };
      });

      const currentFrame = computed(() => {
        if (!viewer.frames.length) return null;
        const idx = Math.min(Math.max(0, viewer.index), viewer.frames.length - 1);
        return viewer.frames[idx];
      });

      const currentContactFrame = computed(() => {
        if (!contactViewer.frames.length) return null;
        const idx = Math.min(Math.max(0, contactViewer.index), contactViewer.frames.length - 1);
        return contactViewer.frames[idx];
      });

      const latestValMse = computed(() => {
        if (!telemetry.vals || !telemetry.vals.length) return null;
        return telemetry.vals[telemetry.vals.length - 1].mse;
      });

      watch(
        () => form.model_size,
        (newSize) => {
          if (form.batch_size_mode === 'Auto') {
            form.batch_size = AUTO_BATCH[newSize] || 8;
          }
        }
      );

      // Parameter linkage watchers: Multi-Resolution ↔ Stage Schedule, Fused Adam ↔ CUDA Graph, Channels ↔ HDR/EV
      watch(
        () => form.multi_resolution,
        (mr) => {
          if (!mr && adv.stage_schedule.enabled) {
            adv.stage_schedule.enabled = false;
          }
        }
      );

      watch(
        () => form.opt_fused_adam,
        (fa) => {
          if (!fa && form.opt_cuda_graph) {
            form.opt_cuda_graph = false;
          }
        }
      );

      watch(
        () => [Number(form.in_channels || 3), Number(form.out_channels || 3)],
        ([ic, oc]) => {
          if (oc === 1 && Number(adv.loss_family.hdr_alpha) !== 0) {
            adv.loss_family.hdr_alpha = 0;
            if (adv.loss_family.enabled) syncLossFamilyEnabled();
          }
          if (ic < 3 && oc < 3 && Number(adv.augment.exposure_ev) !== 0) {
            adv.augment.exposure_ev = 0;
            if (adv.augment.enabled) syncAugmentEnabled();
          }
        }
      );

      const extendEstimatedSteps = computed(() => {
        const curTask = activeTask.value;
        const newEp = Math.max(1, Math.floor(Number(extendTargetEpochs.value) || 0));
        if (!curTask || !newEp) return 0;
        const curEp = Number((curTask.config && curTask.config.train && curTask.config.train.epochs) || 0);
        if (curEp > 0 && curTask.total_steps > 0) {
          return Math.max(1, Math.ceil((newEp * curTask.total_steps) / curEp));
        }
        return newEp;
      });

      // Only initialize extendTargetEpochs when switching tasks or when saved epochs change
      watch(
        () => {
          const curTask = activeTask.value;
          if (!curTask) return '';
          const ep = Number((curTask.config && curTask.config.train && curTask.config.train.epochs) || 0);
          return `${curTask.id}:${ep}`;
        },
        (sig, prevSig) => {
          if (!sig || sig === prevSig) return;
          const curTask = activeTask.value;
          if (!curTask) return;
          if (!extendInputFocused.value) {
            const ep = Number((curTask.config && curTask.config.train && curTask.config.train.epochs) || 0);
            let targetEp = ep > 0 ? ep * 2 : 20000;
            if (userMaxEpochs.value > 0 && targetEp > userMaxEpochs.value) {
              targetEp = userMaxEpochs.value;
            }
            extendTargetEpochs.value = targetEp;
          }
        },
        { immediate: true }
      );

      // ───────── HTTP Helper ─────────
      function _shouldSendAdminHeader(url) {
        if (!adminToken.value) return false;
        const u = String(url || '');
        return u.includes('/admin') || u.includes('all_users=true') || topNav.value === 'admin';
      }

      async function apiFetch(url, options = {}) {
        const headers = Object.assign({}, options.headers || {});
        if (currentLang.value) {
          headers['X-WildCat-Lang'] = currentLang.value;
        }
        if (clientFingerprint) {
          headers['X-Client-FP'] = clientFingerprint;
        }
        const safeRoam = (roamCodeLocal.value || '').replace(/[^\x20-\x7E]/g, '').trim();
        if (safeRoam) {
          headers['X-Roam-Code'] = safeRoam;
        }
        if (_shouldSendAdminHeader(url)) {
          headers['X-Admin-Token'] = adminToken.value;
        }
        if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
          headers['Content-Type'] = 'application/json';
        }
        const res = await fetch(url, { credentials: 'same-origin', ...options, headers });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          if (res.status === 403 && url.includes('/admin')) {
            adminToken.value = '';
            if (typeof sessionStorage !== 'undefined') {
              sessionStorage.removeItem('wc_admin_token');
            }
          }
          const errMsg = Array.isArray(data.detail)
            ? data.detail.map((x) => x.msg || JSON.stringify(x)).join('; ')
            : typeof data.detail === 'object' && data.detail !== null
            ? JSON.stringify(data.detail)
            : data.detail;
          throw new Error(errMsg || t('msg.reqFailedStatus', { status: res.status }));
        }
        return data;
      }

      // ───────── Session, Roaming Passcode & Admin Auth Actions ─────────
      let _sessionEpoch = 0;
      let _fetchTasksInFlight = null;

      function _applyUserSession(u) {
        if (!u) return;
        const switchedUser = !user.value || user.value.id !== u.id;
        user.value = u;
        apiKey.value = u.api_key || '';
        const rc = String(u.recovery_code || '').trim();
        roamCodeLocal.value = rc;
        if (switchedUser) {
          _sessionEpoch += 1;
          _fetchTasksInFlight = null;
        }
        if (typeof localStorage !== 'undefined') {
          if (rc) localStorage.setItem('wc_roam_code', rc);
          else localStorage.removeItem('wc_roam_code');
          localStorage.removeItem('wc_machine_name');
        }
      }

      async function refreshMeQuietly() {
        try {
          const d = await apiFetch('/api/v1/auth/me');
          if (d && d.user) _applyUserSession(d.user);
        } catch (_) {}
      }

      async function restoreSession() {
        try {
          const d = await apiFetch('/api/v1/auth/me');
          if (d.version && d.version.build_label) {
            buildLabel.value = d.version.build_label;
          }
          apiDocsOpen.value = !!d.api_docs;
          _applyUserSession(d.user);
          await initWorkspace();
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          workspaceReady.value = true;
        }
      }

      function openRoamModal() {
        roamModal.errorMsg = '';
        roamModal.codeInput = '';
        roamModal.open = true;
      }

      async function copyTextValue(val, toastKey = '') {
        const text = String(val || '').trim();
        if (!text) return false;
        let ok = false;
        try {
          if (typeof navigator !== 'undefined' && navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            ok = true;
          }
        } catch (_) {}
        if (!ok && typeof document !== 'undefined') {
          try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.focus();
            ta.select();
            ok = document.execCommand('copy');
            document.body.removeChild(ta);
          } catch (_) {}
        }
        if (ok) {
          if (toastKey) notify(t(toastKey), 'success');
        } else {
          notify(t('msg.copyFailedManual'), 'warning');
        }
        return ok;
      }

      async function copyMyRecoveryCode() {
        if (!user.value || !user.value.recovery_code) return;
        await copyTextValue(user.value.recovery_code, 'msg.roamCodeCopiedToast');
      }

      function downloadMyRecoveryTxt() {
        if (!user.value || !user.value.recovery_code) return;
        const lines = [
          'WildCat Workspace Roaming Passcode',
          '==================================',
          `Username (Read-Only): ${user.value.username}`,
          `Roaming Recovery Code: ${user.value.recovery_code}`,
          `Saved At: ${new Date().toISOString()}`,
          '',
          'Keep this code safe. Paste it into "Roaming ID / Cross-Device Recovery" to restore your datasets and training runs on another computer.',
        ];
        const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `WildCat_Roaming_${user.value.username}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 4000);
        notify(t('msg.roamTxtDownloadedToast'), 'success');
      }

      async function submitRecoverWorkspace() {
        roamModal.errorMsg = '';
        const raw = (roamModal.codeInput || '').trim();
        if (!raw) {
          roamModal.errorMsg = t('msg.errRoamCodeEmpty');
          return;
        }
        roamModal.loading = true;
        try {
          const d = await apiFetch('/api/v1/auth/recover', {
            method: 'POST',
            body: JSON.stringify({ recovery_code: raw }),
          });
          _applyUserSession(d.user);
          selectedDatasetId.value = null;
          selectedTaskId.value = null;
          roamModal.open = false;
          roamModal.codeInput = '';
          await initWorkspace();
          notify(t('msg.roamRecoveredToast', { user: d.user.username }), 'success');
        } catch (e) {
          roamModal.errorMsg = e.message;
        } finally {
          roamModal.loading = false;
        }
      }

      async function doAdminLogin() {
        adminAuthError.value = '';
        adminAuthLoading.value = true;
        try {
          const d = await apiFetch('/api/v1/admin/login', {
            method: 'POST',
            body: JSON.stringify({ password: adminPwdInput.value }),
          });
          adminToken.value = d.admin_token;
          if (typeof sessionStorage !== 'undefined') {
            sessionStorage.setItem('wc_admin_token', d.admin_token);
          }
          adminPwdInput.value = '';
          await loadAdminData();
        } catch (e) {
          adminAuthError.value = e.message;
        } finally {
          adminAuthLoading.value = false;
        }
      }

      function logoutAdmin() {
        adminToken.value = '';
        if (typeof sessionStorage !== 'undefined') {
          sessionStorage.removeItem('wc_admin_token');
        }
        goBackToStudio();
      }

      async function changeAdminPassword() {
        if (!pwdForm.old_password || !pwdForm.new_password) {
          notify(t('msg.enterOldAndNewPwd'), 'error');
          return;
        }
        if (pwdForm.new_password.length < 8) {
          notify(t('server.adminPasswordMinLen'), 'error');
          return;
        }
        try {
          const d = await apiFetch('/api/v1/admin/password', {
            method: 'PUT',
            body: JSON.stringify(pwdForm),
          });
          adminToken.value = d.admin_token;
          if (typeof sessionStorage !== 'undefined') {
            sessionStorage.setItem('wc_admin_token', d.admin_token);
          }
          pwdForm.old_password = '';
          pwdForm.new_password = '';
          notify(d.message, 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      // ───────── Workspace Data Loading ─────────
      async function initWorkspace() {
        await Promise.all([
          fetchQueueAndTasks(true),
          fetchAssets(),
          fetchCheckpoints(),
        ]);
        proposeNextTaskName();

        const runningTask = tasks.value.find((x) => x.status === 'running' || x.status === 'queued');
        if (runningTask) {
          selectedTaskId.value = runningTask.id;
          studioMode.value = 'monitor';
          await fetchTelemetry(true);
        } else {
          if (tasks.value.length && !selectedTaskId.value) {
            selectedTaskId.value = tasks.value[0].id;
          }
          studioMode.value = 'create';
        }

        if (typeof window !== 'undefined' && window.location && window.location.pathname === '/admin') {
          topNav.value = 'admin';
          if (adminToken.value) {
            await loadAdminData();
          }
        }
      }

      async function fetchQueueAndTasks(force = false) {
        if (!user.value) return;
        if (!force && _fetchTasksInFlight) return _fetchTasksInFlight;
        const reqEpoch = _sessionEpoch;
        const run = (async () => {
          const [qSettled, tSettled] = await Promise.allSettled([
            apiFetch('/api/v1/queue'),
            apiFetch('/api/v1/tasks'),
          ]);
          if (reqEpoch !== _sessionEpoch) return;
          if (qSettled.status === 'fulfilled' && qSettled.value && qSettled.value.queue) {
            Object.assign(queue, qSettled.value.queue);
            if (form.preferred_gpu && form.preferred_gpu !== 'auto') {
              const selGpu = (queue.gpus || []).find((g) => String(g.gpu_id) === String(form.preferred_gpu));
              if (selGpu && selGpu.enabled === false) {
                form.preferred_gpu = 'auto';
              }
            }
            if (Array.isArray(queue.gpus) && queue.gpus.length) {
              adminSettings.gpu_pool.gpus = queue.gpus;
              adminSettings.gpu_pool.total_gpu_count = Number(queue.total_gpu_count ?? queue.gpus.length);
              adminSettings.gpu_pool.enabled_gpu_count = Number(
                queue.enabled_gpu_count ?? queue.gpus.filter((g) => g.enabled !== false).length
              );
              adminSettings.gpu_pool.all_gpus_disabled = !!queue.all_gpus_disabled;
              adminSettings.gpu_pool.primary_gpu = queue.primary_gpu ?? null;
              adminSettings.gpu_pool.secondary_gpu = queue.secondary_gpu ?? null;
              adminSettings.gpu_pool.disabled_gpus = queue.gpus
                .filter((g) => g.enabled === false)
                .map((g) => String(g.gpu_id));
            }
          }
          if (tSettled.status === 'fulfilled' && tSettled.value) {
            const tRes = tSettled.value;
            tasks.value = tRes.tasks || [];
            // temporary users: the limit counts the tasks in progress (queued + training); a finished task frees its
            // place at once, so the header and the submit check follow the task list instead of the last login
            if (user.value && !isPrivilegedUser.value) {
              const uid = user.value.id;
              user.value.tasks_used = tasks.value.filter(
                (x) => (x.user_id == null || x.user_id === uid) && (x.status === 'queued' || x.status === 'running')
              ).length;
            }
            if (tRes.lab_groups) {
              applyServerLabGroups(tRes.lab_groups);
            }
            if (tRes.db_summary) {
              Object.assign(labDbSummary, tRes.db_summary);
            }
            if (tasks.value.length && (!selectedTaskId.value || !tasks.value.some((x) => x.id === selectedTaskId.value))) {
              selectedTaskId.value = tasks.value[0].id;
            }
          } else if (tSettled.status === 'rejected') {
            console.warn('[WildCat] fetch /api/v1/tasks failed:', tSettled.reason);
          }
        })();
        _fetchTasksInFlight = run;
        try {
          await run;
        } finally {
          if (_fetchTasksInFlight === run) {
            _fetchTasksInFlight = null;
          }
        }
      }

      async function fetchAssets() {
        if (!user.value) return;
        try {
          const d = await apiFetch('/api/v1/assets');
          assets.value = d.assets || [];
          if (
            pairedAssets.value.length &&
            (!selectedDatasetId.value || !pairedAssets.value.some((a) => a.id === selectedDatasetId.value))
          ) {
            selectedDatasetId.value = pairedAssets.value[0].id;
          } else if (!pairedAssets.value.length) {
            selectedDatasetId.value = null;
          }
          if (labEvalAssetId.value && !assets.value.some((a) => a.id === labEvalAssetId.value)) {
            labEvalAssetId.value = null;
          }
          syncFormWithSelectedDataset();
          if (selectedDataset.value && selectedDataset.value.id) {
            ensureAssetFrames(selectedDataset.value.id);
          }
        } catch (_) {}
      }

      function syncFormWithSelectedDataset() {
        const ds = selectedDataset.value;
        if (!ds) {
          form.source = '';
          form.target = '';
          return;
        }
        form.source = ds.source_path || ds.rel_path || '';
        form.target = ds.target_path || ds.rel_path || '';
        if (!isChannelsLocked.value && ds.in_channels && ds.out_channels) {
          const ic = Number(ds.in_channels || 3);
          const oc = Number(ds.out_channels || 3);
          if (oc === 1) {
            syncLockedChannels(ic >= 4 ? 4 : 3, 1);
          } else if (ic >= 4 && oc >= 4) {
            syncLockedChannels(4, 4);
          } else if (ic === 3 && oc >= 4) {
            syncLockedChannels(3, 4);
          } else {
            syncLockedChannels(3, 3);
          }
        }
      }

      function onSelectDataset() {
        syncFormWithSelectedDataset();
        if (selectedDataset.value && selectedDataset.value.id) {
          ensureAssetFrames(selectedDataset.value.id);
        }
      }

      async function fetchCheckpoints() {
        if (!user.value) return;
        try {
          const d = await apiFetch('/api/v1/checkpoints');
          rawCheckpoints.value = (d.checkpoints || []).filter((c) => c.path);
        } catch (_) {}
      }

      function proposeNextTaskName() {
        const existing = new Set(tasks.value.map((x) => x.node_name));
        const prefix = user.value && user.value.username !== 'admin' ? `${user.value.username}_WildCat` : 'WildCat';
        let i = 1;
        while (existing.has(`${prefix}${i}`)) i++;
        form.run_name = `${prefix}${i}`;
      }

      function openCreateWizard() {
        proposeNextTaskName();
        topNav.value = 'studio';
        studioMode.value = 'create';
        fetchCheckpoints();
      }

      function selectTaskForMonitor(taskObj) {
        const switchedTask = selectedTaskId.value !== taskObj.id;
        selectedTaskId.value = taskObj.id;
        topNav.value = 'studio';
        studioMode.value = 'monitor';
        viewer.selectedPair = null;
        viewer.playing = false;
        contactViewer.playing = false;
        if (switchedTask) {
          telemetry.steps = [];
          telemetry.vals = [];
          telemetry.tail = [];
          telemetry.latest = {};
          viewer.pairs = [];
          viewer.frames = [];
          viewer.index = 0;
          contactViewer.frames = [];
          contactViewer.index = 0;
          taskKfCommon.value = null;
          taskQuality.value = null;
          qualityUi.hoverPoint = -1;
          qualityUi.hoverFrame = -1;
        }
        taskModels.has_eval_sequence = false;
        resetViewerZoom();
        resetContactZoom();
        fetchTelemetry(true);
      }

      // ───────── Initial Weights & Strict Model Structure Locking ─────────
      function syncLockedChannels(inCh, outCh) {
        form.in_channels = Number(inCh || 3);
        form.out_channels = Number(outCh || 3);
        channelPreset.value = `${form.in_channels}->${form.out_channels}`;
      }

      function isChannelPresetAllowed(preset) {
        const ds = selectedDataset.value;
        if (!ds || !ds.in_channels || !ds.out_channels) return true;
        const [reqInStr, reqOutStr] = String(preset || '3->3').split('->');
        const reqIn = Number(reqInStr || 3);
        const reqOut = Number(reqOutStr || 3);
        const dsIn = Number(ds.in_channels || 3);
        const dsOut = Number(ds.out_channels || 3);
        if (dsIn < reqIn) return false;
        if (reqOut > 1 && dsOut < reqOut) return false;
        return true;
      }

      function channelOptionLabel(preset) {
        const keyMap = {
          '3->3': 'wizard.channelPreset33',
          '3->4': 'wizard.channelPreset34',
          '4->4': 'wizard.channelPreset44',
          '3->1': 'wizard.channelPreset31',
          '4->1': 'wizard.channelPreset41',
        };
        const base = t(keyMap[preset] || 'wizard.channelPreset33');
        if (isChannelsLocked.value || isChannelPresetAllowed(preset) || !selectedDataset.value) {
          return base;
        }
        return (
          base +
          t('wizard.channelUnavailableHint', {
            inCh: selectedDataset.value.in_channels || 3,
            outCh: selectedDataset.value.out_channels || 3,
          })
        );
      }

      function onChannelPresetChange() {
        if (!isChannelsLocked.value && !isChannelPresetAllowed(channelPreset.value)) {
          syncFormWithSelectedDataset();
          return;
        }
        const [ic, oc] = String(channelPreset.value || '3->3').split('->');
        form.in_channels = Number(ic || 3);
        form.out_channels = Number(oc || 3);
      }


      function onCheckpointSelected() {
        const meta = selectedCkptMeta.value;
        if (meta && meta.path) {
          form.model_size = meta.model_size || 'Medium';
          syncLockedChannels(meta.in_channels || 3, meta.out_channels || 3);
        }
      }

      function applyCkptPaths() {
        const meta = selectedCkptMeta.value;
        if (!meta) return;
        const norm = (s) => String(s || '').replace(/\\/g, '/');
        const mSrc = norm(meta.source);
        const match = assets.value.find(
          (a) =>
            mSrc &&
            (mSrc.startsWith(norm(a.source_path)) || norm(a.source_path).startsWith(mSrc.replace('/source.####.png', '')))
        );
        if (match) {
          selectedDatasetId.value = match.id;
          syncFormWithSelectedDataset();
        } else {
          if (meta.source) form.source = meta.source;
          if (meta.target) form.target = meta.target;
        }
      }

      function selectInitCategory(cat) {
        initCategory.value = cat;
        if (cat === 'scratch') {
          form.initial_weights_mode = 'None';
          form.checkpoint_path = '';
          form.model_size = 'Medium';
          if (selectedDataset.value) {
            syncFormWithSelectedDataset();
          } else {
            syncLockedChannels(3, 3);
          }
        } else if (cat === 'checkpoint') {
          form.initial_weights_mode = 'Checkpoint';
          if (!form.checkpoint_path && checkpoints.value.length && checkpoints.value[0].path) {
            form.checkpoint_path = checkpoints.value[0].path;
          }
          onCheckpointSelected();
        }
      }

      // ───────── Form & Hyperparameter Logic ─────────
      function onModelSizeChange() {
        if (form.batch_size_mode === 'Auto') {
          form.batch_size = AUTO_BATCH[form.model_size] || 8;
        }
      }

      function onBsModeChange() {
        if (form.batch_size_mode === 'Auto') {
          form.batch_size = AUTO_BATCH[form.model_size] || 8;
        } else {
          form.batch_size = Math.max(1, Math.floor(Number(form.batch_size) || 8));
        }
      }

      // One master switch for all acceleration options (all lossless; fused_adam set before cuda_graph)
      const ACCEL_KEYS = ['opt_gpu_resident', 'opt_fused_adam', 'opt_cuda_graph', 'opt_channels_last', 'opt_cudnn_benchmark', 'opt_async_io'];
      const accelAll = computed({
        get: () => ACCEL_KEYS.every((k) => !!form[k]),
        set: (v) => ACCEL_KEYS.forEach((k) => { form[k] = !!v; }),
      });
      const accelAny = computed(() => ACCEL_KEYS.some((k) => !!form[k]));

      function formatGpuOptionLabel(g) {
        if (!g) return '';
        const prefix = g.is_primary
          ? t('wizard.gpuRolePrimaryPrefix')
          : g.is_secondary
            ? t('wizard.gpuRoleSecondaryPrefix')
            : '';
        const fullLabel = `${prefix}${g.label || ('CUDA:' + g.gpu_id)}`;
        if (g.enabled === false) {
          return t('wizard.gpuDisabledOpt', { label: fullLabel });
        }
        if (g.full) {
          return t('wizard.gpuBusyOpt', { label: fullLabel, eta: formatEta(g.gpu_wait_sec) });
        }
        return t('wizard.gpuIdleOpt', { label: fullLabel });
      }

      const activeEtaGpuProfile = computed(() => {
        const dbObj = gpuBenchDb.value;
        const list = (dbObj && Array.isArray(dbObj.detected_gpus) && dbObj.detected_gpus) || [];
        if (!list.length) return null;
        const qObjMap = {};
        for (const qg of (queue.gpus || [])) {
          qObjMap[String(qg.gpu_id)] = qg;
        }
        const pref = String(form.preferred_gpu || 'auto');
        if (pref !== 'auto') {
          const hit = list.find((g) => String(g.gpu_id) === pref);
          const qHit = qObjMap[pref];
          if (hit && (!qHit || qHit.enabled !== false)) return hit;
        }
        const enabledList = list.filter((g) => {
          const qg = qObjMap[String(g.gpu_id)];
          return !qg || qg.enabled !== false;
        });
        const pool = enabledList.length ? enabledList : list;
        const sorted = [...pool].sort((a, b) => {
          const qa = qObjMap[String(a.gpu_id)];
          const qb = qObjMap[String(b.gpu_id)];
          const wa = Number(qa?.gpu_wait_sec || 0);
          const wb = Number(qb?.gpu_wait_sec || 0);
          if (wa !== wb) return wa - wb;
          const ra = qa?.is_primary ? 0 : qa?.is_secondary ? 1 : 2;
          const rb = qb?.is_primary ? 0 : qb?.is_secondary ? 1 : 2;
          if (ra !== rb) return ra - rb;
          return Number(b.vram_mb || 0) - Number(a.vram_mb || 0);
        });
        return sorted[0] || list[0];
      });

      function estimateConfigDurationSec(cfg) {
        const cObj = cfg || {};
        const mo = cObj.model || {};
        const tr = cObj.train || {};
        const av = cObj.advanced || {};
        const io = cObj.io || {};
        const opt = cObj.optimizations || {};
        const tot = Math.max(1, Number(tr.total_steps) || 1);
        const msize = mo.model_size || 'Medium';
        const bs = Math.max(1, Number(tr.batch_size) || 8);
        const crop = Math.max(32, Number(tr.crop_size) || 256);
        const isFastEngine = String(opt.engine || 'fast') !== 'reference';
        const prof = activeEtaGpuProfile.value;

        let b4, b2, b1, warmupSec = 7.5;
        if (prof && prof.curves && prof.curves[msize]) {
          const c = prof.curves[msize];
          const buckets = prof.buckets || {};
          const aloss = av.loss_family && av.loss_family.enabled ? 1 : 0;
          const emult = av.ema_lr && av.ema_lr.enabled ? Number(c.adv_ema_mult || 1.015) : 1.0; // one extra blend per parameter per step
          const gmult = av.augment && av.augment.enabled ? Number(c.adv_aug_mult || 1.015) : 1.0;
          const rmult = !isFastEngine ? Number(c.ref_engine_mult || 2.15) : 1.0;
          const lmult = aloss ? Number(c.adv_loss_mult || 1.065) : 1.0;
          const tl = Number(c.tau_launch_ms || 2.5);
          const tp = Number(c.tau_pix_ms || 2.2);
          const bexp = Number(c.batch_exp || 0.90);
          const aexp = Number(c.area_exp || 0.98);
          warmupSec = Number(c.warmup_sec || 7.5);

          const calcSps = (sc) => {
            const spatial = Math.pow((crop / sc) / 256.0, 2.0 * aexp);
            const tMs = (tl + tp * Math.pow(bs, bexp) * spatial) * lmult;
            const curveSps = 1000.0 / Math.max(0.2, tMs);
            const bHit = buckets[`${msize}|${crop}|${bs}|${aloss}|${sc}`];
            const blended = bHit != null && bHit > 0 ? (0.75 * Number(bHit) + 0.25 * curveSps) : curveSps;
            return Math.max(1.0, blended / (emult * gmult * rmult));
          };
          b4 = calcSps(4);
          b2 = calcSps(2);
          b1 = calcSps(1);
        } else {
          [b4, b2, b1] = msize === 'Small' ? [450, 265, 105] : (msize === 'Large' ? [134, 70, 26] : [295, 158, 62]);
          if (bs !== 8) {
            const f = Math.pow(8 / bs, 0.72);
            b4 *= f; b2 *= f; b1 *= f;
          }
          if (crop !== 256) {
            const r = 256 / crop;
            b4 *= Math.pow(r, 0.85);
            b2 *= Math.pow(r, 1.35);
            b1 *= Math.pow(r, 1.85);
          }
          if (!isFastEngine) {
            b4 *= 0.40; b2 *= 0.45; b1 *= 0.55;
          }
        }

        let sec = 0;
        if (tr.multi_resolution !== false) {
          let s1, s2;
          if (av.stage_schedule && av.stage_schedule.enabled) {
            const rq = Math.max(0, Number(av.stage_schedule.r_quarter ?? 15));
            const rh = Math.max(0, Number(av.stage_schedule.r_half ?? 25));
            const rf = Math.max(0, Number(av.stage_schedule.r_full ?? 60));
            const rSum = Math.max(1e-6, rq + rh + rf);
            s1 = Math.min(tot, Math.max(0, Math.round((tot * rq) / rSum)));
            s2 = Math.min(tot, Math.max(s1, Math.round((tot * (rq + rh)) / rSum)));
          } else {
            const s = Math.ceil(tot / 3);
            s1 = Math.min(tot, s);
            s2 = Math.min(tot, 2 * s);
          }
          sec = s1 / b4 + Math.max(0, s2 - s1) / b2 + Math.max(0, tot - s2) / b1;
        } else {
          sec = tot / b1;
        }
        const ckpts = Math.max(1, Math.floor(tot / Math.max(10, Number(io.checkpoint_interval) || 1000)));
        return Math.max(2, Math.round(sec + ckpts * 0.12 + warmupSec));
      }

      const estimatedNewTaskDurationSec = computed(() => {
        return estimateConfigDurationSec({
          model: { model_size: form.model_size || 'Medium' },
          train: {
            total_steps: Math.max(1, Number(computedTotalSteps.value) || 1),
            batch_size: Math.max(1, Number(effectiveBatchSize.value) || 8),
            crop_size: Math.max(32, Number(form.crop_size) || 256),
            multi_resolution: !!form.multi_resolution,
          },
          advanced: {
            loss_family: { enabled: isAdvModuleEffective('loss_family') },
            augment: { enabled: isAdvModuleEffective('augment') },
            ema_lr: { enabled: isAdvModuleEffective('ema_lr') },
            stage_schedule: {
              enabled: isAdvModuleEffective('stage_schedule'),
              r_quarter: adv.stage_schedule.r_quarter,
              r_half: adv.stage_schedule.r_half,
              r_full: adv.stage_schedule.r_full,
            },
          },
          optimizations: { engine: accelAny.value ? 'fast' : 'reference' },
          io: { checkpoint_interval: form.checkpoint_interval },
        });
      });

      const wizardQueuePreview = computed(() => {
        const pref = String(form.preferred_gpu || 'auto');
        const qCount = Number(queue.queued_count || 0);
        if (pref === 'auto') {
          const willQueue = Number(queue.idle_gpu_count ?? 1) <= 0 && (queue.gpus || []).length > 0;
          const waitSec = willQueue ? Number(queue.next_auto_wait_sec || 0) : 0;
          const aheadCount = willQueue ? (Number(queue.running_count || 0) + qCount) : 0;
          return { willQueue, waitSec, aheadCount, nextQueuePos: qCount + 1 };
        }
        const g = (queue.gpus || []).find((x) => String(x.gpu_id) === pref);
        const willQueue = g ? !!g.full : false;
        const waitSec = willQueue ? Number(g.gpu_wait_sec || 0) : 0;
        const pinnedQueued = (queue.queued || []).filter((q) => String(q.target_gpu) === pref).length;
        const aheadCount = willQueue ? (Number(g.active_count || 1) + pinnedQueued) : 0;
        return { willQueue, waitSec, aheadCount, nextQueuePos: qCount + 1 };
      });

      function buildConfigPayload(resume = false) {
        let iw = 'None';
        if (initCategory.value === 'checkpoint') {
          iw = form.checkpoint_path || '';
        }
        const payload = {
          run_name: (form.run_name || 'WildCat1').trim(),
          paths: {
            source: (form.source || '').trim(),
            target: (form.target || '').trim(),
          },
          color: {
            working_colorspace: form.working_colorspace || 'ACEScg',
            source_colorspace: 'auto',
            target_colorspace: 'auto',
            output_colorspace: form.working_colorspace || 'ACEScg',
            output_format: form.output_format || 'exr_half',
            view_transform: 'ACES 1.0 - SDR Video',
          },
          model: {
            model_size: form.model_size,
            in_channels: Number(form.in_channels || 3),
            out_channels: Number(form.out_channels || 3),
            n_levels: 3,
            scale: 0.5,
          },
          train: {
            epochs: Number(form.epochs || 10000),
            total_steps: computedTotalSteps.value,
            batch_size_auto: form.batch_size_mode === 'Auto',
            batch_size: form.batch_size_mode === 'Auto' ? 8 : Math.max(1, Math.floor(Number(form.batch_size) || 8)),
            crop_size: Number(form.crop_size || 256),
            learning_rate: Number(form.learning_rate || 0.0001),
            lr_power: 0.3,
            multi_resolution: !!form.multi_resolution,
            precision: form.precision || 'adaptive',
            seed: Number(form.seed || 0),
            initial_weights: iw,
            resume: !!resume,
          },
          optimizations: {
            engine: accelAny.value ? 'fast' : 'reference',
            lossless: {
              gpu_resident: form.opt_gpu_resident ? 'auto' : false,
              cuda_graph: !!(form.opt_cuda_graph && form.opt_fused_adam),
              fused_adam: !!form.opt_fused_adam,
              channels_last: !!form.opt_channels_last,
              cudnn_benchmark: !!form.opt_cudnn_benchmark,
              async_io: !!form.opt_async_io,
              preview_gpu: form.preview_gpu || 'auto',
            },
          },
          loss: { l1_weight: 0.0, l2_weight: 1.0 },
          io: {
            checkpoint_interval: Math.max(10, Number(form.checkpoint_interval || 1000)),
            preview_enabled: form.preview_gpu !== 'off',
            preview_interval: Math.max(10, Number(form.preview_interval || 100)),
            preview_size: 0,
            log_interval: Math.max(1, Number(form.log_interval || 10)),
          },
        };
        if (advActiveCount.value > 0) {
          const cleanAdv = JSON.parse(JSON.stringify(adv));
          for (const k of ADV_KEYS) {
            if (cleanAdv[k]) cleanAdv[k].enabled = isAdvModuleEffective(k);
          }
          if (Number(form.out_channels) === 1 && cleanAdv.loss_family) {
            cleanAdv.loss_family.hdr_alpha = 0;
          }
          if (Number(form.in_channels) < 3 && Number(form.out_channels) < 3 && cleanAdv.augment) {
            cleanAdv.augment.exposure_ev = 0;
          }
          if (cleanAdv.ema_lr) {
            const d = Number(cleanAdv.ema_lr.ema_decay);
            cleanAdv.ema_lr.ema_decay = Number.isFinite(d) ? Math.max(0.9, Math.min(0.99999, d)) : 0.9995; // same clamp as wildcat/advanced.py
          }
          if (cleanAdv.kf_common) {
            Object.assign(cleanAdv.kf_common, clampKfCommon(cleanAdv.kf_common));
          }
          payload.advanced = cleanAdv;
        }
        if (selectedDataset.value && selectedDataset.value.id) {
          payload.experiment = Object.assign({}, payload.experiment || {}, {
            dataset_id: selectedDataset.value.id,
          });
          if (selectedDataset.value.has_eval_sequence && selectedDataset.value.seq_path) {
            payload.paths.eval_sequence = selectedDataset.value.seq_path;
          }
        }
        return payload;
      }

      function taskAheadCount(taskObj) {
        if (!taskObj || taskObj.status !== 'queued') return 0;
        if (taskObj.ahead_count != null) return Number(taskObj.ahead_count);
        return Number(queue.running_count || 0) + Math.max(0, Number(taskObj.queue_position || 1) - 1);
      }

      function formatEta(sec) {
        if (sec == null || Number.isNaN(Number(sec))) return t('msg.etaCalculating');
        const s = Math.max(0, Math.round(Number(sec)));
        if (s <= 0) return t('msg.etaStartingSoon');
        if (s < 60) return t('msg.etaSec', { s });
        const h = Math.floor(s / 3600);
        const m = Math.floor((s % 3600) / 60);
        const remS = s % 60;
        if (h > 0) {
          return m > 0 ? t('msg.etaHourMin', { h, m }) : t('msg.etaHour', { h });
        }
        if (m < 10 && remS > 0) {
          return t('msg.etaMinSec', { m, s: remS });
        }
        return t('msg.etaMin', { m });
      }

      function formatQueuedToast(taskObj, prefixMsg) {
        const pos = taskObj.queue_position || 1;
        const ahead = taskAheadCount(taskObj);
        const waitStr = formatEta(taskObj.estimated_wait_sec);
        return t('msg.queuedToast', { prefix: prefixMsg, pos, ahead, wait: waitStr });
      }

      // ───────── 1. Launch Brand-New Training Task (Starts at Step 0) ─────────
      async function submitNewTrainingTask() {
        if (userTaskQuotaReached.value && user.value) {
          notify(
            t('msg.errTempMaxTasks', {
              max: user.value.max_tasks ?? 2,
              used: user.value.tasks_used ?? 0,
            }),
            'error'
          );
          return;
        }
        if (!prepareFormForStaging()) return;
        try {
          const cfg = buildConfigPayload(false);
          const res = await apiFetch('/api/v1/tasks', {
            method: 'POST',
            body: JSON.stringify({
              node_name: cfg.run_name,
              job_type: 'train',
              preferred_gpu: form.preferred_gpu || 'auto',
              config: cfg,
            }),
          });
          await Promise.all([fetchQueueAndTasks(), fetchCheckpoints(), refreshMeQuietly()]);
          selectedTaskId.value = res.task.id;
          studioMode.value = 'monitor';
          proposeNextTaskName();
          const fresh = tasks.value.find((x) => x.id === res.task.id) || res.task;
          if (fresh.status === 'running') {
            notify(t('msg.newTaskStartedGpu', { node: fresh.node_name, gpu: fresh.assigned_gpu }), 'success');
          } else {
            notify(formatQueuedToast(fresh, t('msg.newTaskSubmittedPrefix', { node: fresh.node_name })), 'warning');
          }
          await fetchTelemetry(true);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      // ───────── 2. Resume an Interrupted / Cancelled Task (Re-queues cleanly) ─────────
      async function resumeExistingTask(taskObj) {
        if (!taskObj) return;
        try {
          const hasSavedCfg = taskObj.config && typeof taskObj.config === 'object' && Object.keys(taskObj.config).length > 0;
          const baseCfg = JSON.parse(JSON.stringify(hasSavedCfg ? taskObj.config : buildConfigPayload(!!taskObj.has_ckpt)));
          baseCfg.run_name = taskObj.node_name;
          baseCfg.train = Object.assign({}, baseCfg.train || {}, {
            resume: !!taskObj.has_ckpt,
          });
          const res = await apiFetch('/api/v1/tasks', {
            method: 'POST',
            body: JSON.stringify({
              node_name: taskObj.node_name,
              job_type: 'train',
              preferred_gpu: taskObj.preferred_gpu || 'auto',
              config: baseCfg,
            }),
          });
          await fetchQueueAndTasks();
          selectedTaskId.value = res.task.id;
          studioMode.value = 'monitor';
          const fresh = tasks.value.find((x) => x.id === res.task.id) || res.task;
          if (fresh.status === 'running') {
            if (taskObj.has_ckpt) {
              notify(
                t('msg.taskResumedFromStep', {
                  node: fresh.node_name,
                  gpu: fresh.assigned_gpu,
                  step: taskObj.last_step || 0,
                }),
                'success'
              );
            } else {
              notify(t('msg.taskStartedGpu', { node: fresh.node_name, gpu: fresh.assigned_gpu }), 'success');
            }
          } else {
            notify(formatQueuedToast(fresh, t('msg.taskQueuedPrefix', { node: fresh.node_name })), 'warning');
          }
          await fetchTelemetry(true);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      // ───────── 3. Extend Target Epochs on an Existing Task (No Cap, Re-queues at Tail) ─────────
      async function extendExistingTask(taskObj) {
        if (!taskObj) return;
        const curEpochs = Number((taskObj.config && taskObj.config.train && taskObj.config.train.epochs) || 0);
        const newEpochs = Math.floor(Number(extendTargetEpochs.value || 0));
        if (!Number.isFinite(newEpochs) || newEpochs < 1) {
          notify(t('msg.errExtendInvalidEpochs'), 'error');
          return;
        }
        if (userMaxEpochs.value > 0 && newEpochs > userMaxEpochs.value) {
          notify(t('msg.errTempMaxEpochs', { max: userMaxEpochs.value, req: newEpochs }), 'error');
          return;
        }
        const stepsPerEpoch = curEpochs > 0 && taskObj.total_steps > 0 ? taskObj.total_steps / curEpochs : 0;
        const newSteps = stepsPerEpoch > 0 ? Math.ceil(newEpochs * stepsPerEpoch) : 0;
        if (curEpochs && newEpochs <= curEpochs && (!newSteps || newSteps <= (taskObj.last_step || 0))) {
          notify(t('msg.errExtendMustExceed', { next: newEpochs, cur: curEpochs }), 'error');
          return;
        }
        if (taskObj.status === 'running' && Number(queue.queued_count || 0) > 0) {
          const ok = confirm(
            t('msg.confirmExtendRequeue', { queued: queue.queued_count, epochs: newEpochs })
          );
          if (!ok) return;
        }
        try {
          const baseCfg = JSON.parse(JSON.stringify(taskObj.config || buildConfigPayload(true)));
          baseCfg.run_name = taskObj.node_name;
          baseCfg.train = Object.assign({}, baseCfg.train || {}, { resume: true, epochs: newEpochs });
          delete baseCfg.train.total_steps;
          baseCfg._requeue_extend = true;
          extendInputFocused.value = false;
          const res = await apiFetch('/api/v1/tasks', {
            method: 'POST',
            body: JSON.stringify({
              node_name: taskObj.node_name,
              job_type: 'train',
              preferred_gpu: taskObj.preferred_gpu || 'auto',
              config: baseCfg,
            }),
          });
          await fetchQueueAndTasks();
          selectedTaskId.value = res.task.id;
          studioMode.value = 'monitor';
          const fresh = tasks.value.find((x) => x.id === res.task.id) || res.task;
          if (fresh.status === 'running') {
            notify(
              t('msg.taskExtendStartedGpu', {
                node: fresh.node_name,
                gpu: fresh.assigned_gpu,
                epochs: newEpochs,
                steps: fresh.total_steps || newSteps || '?',
              }),
              'success'
            );
          } else {
            notify(
              formatQueuedToast(fresh, t('msg.taskExtendQueuedPrefix', { node: fresh.node_name, epochs: newEpochs })),
              'warning'
            );
          }
          await fetchTelemetry(true);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      function retryFailedTask(taskObj) {
        if (!taskObj) return;
        const cfg = taskObj.config || {};
        const paths = cfg.paths || {};
        const tr = cfg.train || {};
        const mo = cfg.model || {};
        form.run_name = taskObj.node_name || form.run_name;
        if (paths.source) form.source = paths.source;
        if (paths.target) form.target = paths.target;
        const norm = (s) => String(s || '').replace(/\\/g, '/');
        const fSrc = norm(form.source);
        const match = assets.value.find(
          (a) => norm(a.source_path) === fSrc || norm(a.rel_path) === fSrc
        );
        if (match) {
          selectedDatasetId.value = match.id;
          syncFormWithSelectedDataset();
        }
        const iw = String(tr.initial_weights || 'None');
        if (!iw || iw === 'None') {
          initCategory.value = 'scratch';
          form.initial_weights_mode = 'None';
          form.checkpoint_path = '';
        } else {
          initCategory.value = 'checkpoint';
          form.initial_weights_mode = 'Checkpoint';
          form.checkpoint_path = iw;
          onCheckpointSelected();
        }
        if (tr.epochs) form.epochs = Number(tr.epochs);
        if (mo.model_size) form.model_size = mo.model_size;
        if (mo.in_channels && mo.out_channels) {
          syncLockedChannels(mo.in_channels, mo.out_channels);
        }
        if (typeof tr.batch_size_auto === 'boolean') {
          form.batch_size_mode = tr.batch_size_auto ? 'Auto' : 'Manual';
        }
        if (tr.batch_size) form.batch_size = Math.max(1, Math.floor(Number(tr.batch_size)));
        if (tr.crop_size) form.crop_size = Number(tr.crop_size);
        if (typeof tr.multi_resolution === 'boolean') form.multi_resolution = tr.multi_resolution;
        if (tr.precision) form.precision = tr.precision;
        if (tr.learning_rate) form.learning_rate = Number(tr.learning_rate);
        if (tr.seed != null) form.seed = Number(tr.seed);
        const ioCfg = cfg.io || {};
        if (ioCfg.checkpoint_interval) form.checkpoint_interval = Number(ioCfg.checkpoint_interval);
        if (ioCfg.preview_interval) form.preview_interval = Number(ioCfg.preview_interval);
        if (ioCfg.log_interval) form.log_interval = Number(ioCfg.log_interval);
        resetAdvToNukeNative(true);
        if (cfg.advanced && typeof cfg.advanced === 'object') {
          for (const k of ADV_KEYS) {
            if (cfg.advanced[k] && typeof cfg.advanced[k] === 'object') {
              Object.assign(adv[k], cfg.advanced[k]);
            }
          }
        }
        syncKfCommonParams();
        topNav.value = 'studio';
        studioMode.value = 'create';
      }

      function taskEpoch(taskObj) {
        const ep = Number((taskObj && taskObj.config && taskObj.config.train && taskObj.config.train.epochs) || 0);
        if (!taskObj || !ep || !taskObj.total_steps) return 0;
        return Math.min(ep, Math.floor(((taskObj.last_step || 0) * ep) / taskObj.total_steps));
      }

      // ───────── Monitor, Telemetry & Loss Chart ─────────
      function switchViewerMode(m) {
        viewer.mode = m;
        fetchTelemetry(true);
      }

      let _telemetryReqSeq = 0;

      async function fetchTelemetry(resetToLatest = false) {
        if (!selectedTaskId.value || !user.value) return;
        const reqTaskId = selectedTaskId.value;
        const reqMode = viewer.mode;
        const reqSeq = ++_telemetryReqSeq;
        try {
          const pairParam = viewer.selectedPair ? `&pair=${encodeURIComponent(viewer.selectedPair)}` : '';
          const d = await apiFetch(`/api/v1/tasks/${reqTaskId}/telemetry?mode=${reqMode}${pairParam}`);
          if (reqSeq !== _telemetryReqSeq || selectedTaskId.value !== reqTaskId || viewer.mode !== reqMode) {
            return;
          }
          const prevTailLen = telemetry.tail.length;
          const prevLastLine = prevTailLen ? telemetry.tail[prevTailLen - 1] : '';
          telemetry.steps = d.steps || [];
          telemetry.vals = d.vals || [];
          telemetry.tail = d.tail || [];
          telemetry.latest = d.latest || {};
          const newLastLine = telemetry.tail.length ? telemetry.tail[telemetry.tail.length - 1] : '';
          const tailChanged = telemetry.tail.length !== prevTailLen || newLastLine !== prevLastLine;

          const idx = tasks.value.findIndex((x) => x.id === d.task.id);
          if (idx >= 0) {
            const prevStatus = tasks.value[idx].status;
            tasks.value[idx] = d.task;
            if (d.task.status === 'failed' && (prevStatus === 'running' || prevStatus === 'queued')) {
              const lastLine = String(d.task.error_msg || '')
                .trim()
                .split('\n')
                .filter(Boolean)
                .slice(-1)[0];
              notify(
                t('msg.taskFailedToast', {
                  node: d.task.node_name,
                  detail: lastLine || t('msg.taskFailedSeeDetails'),
                }),
                'error'
              );
            }
          }

          // 1. Full-Frame Sequence Viewer
          const vw = d.viewer || {};
          viewer.pairs = vw.pairs || [];
          viewer.selectedPair = vw.selected_pair || null;
          const prevLen = viewer.frames.length;
          const wasAtViewerEnd = prevLen === 0 || viewer.index >= prevLen - 1;
          viewer.frames = vw.frames || [];
          if (resetToLatest || (viewer.mode === 'preview' && viewer.frames.length > prevLen && !viewer.playing && wasAtViewerEnd)) {
            viewer.index = Math.max(0, viewer.frames.length - 1);
          } else if (viewer.index >= viewer.frames.length) {
            viewer.index = Math.max(0, viewer.frames.length - 1);
          }

          // 2. Nuke 3x3 Contact Sheet Viewer
          const cs = d.contact_sheet || {};
          const prevCsLen = contactViewer.frames.length;
          const wasAtCsEnd = prevCsLen === 0 || contactViewer.index >= prevCsLen - 1;
          contactViewer.cropSize = Number(cs.crop_size || 256);
          contactViewer.nukePrefix = cs.nuke_prefix || '';
          contactViewer.frames = cs.frames || [];
          if (resetToLatest || (contactViewer.frames.length > prevCsLen && !contactViewer.playing && wasAtCsEnd)) {
            contactViewer.index = Math.max(0, contactViewer.frames.length - 1);
          } else if (contactViewer.index >= contactViewer.frames.length) {
            contactViewer.index = Math.max(0, contactViewer.frames.length - 1);
          }

          // 3. Task Model Versions Inventory (Nuke Naming)
          const md = d.models || {};
          taskModels.nuke_prefix = md.nuke_prefix || '';
          taskModels.zip_filename = md.zip_filename || (md.nuke_prefix ? `${md.nuke_prefix}.zip` : '');
          taskModels.count = Number(md.count || 0);
          taskModels.items = md.items || [];
          taskModels.has_eval_sequence = !!md.has_eval_sequence;

          // 4. Keyframe commonality report (advanced.kf_common), tagged with the task it belongs to
          taskKfCommon.value =
            d.kf_common && typeof d.kf_common === 'object' ? Object.assign({ task_id: reqTaskId }, d.kf_common) : null;

          // 5. Training quality on the in-between frames (field "quality"), tagged with the task it belongs to
          taskQuality.value =
            d.quality && typeof d.quality === 'object' ? Object.assign({ task_id: reqTaskId }, d.quality) : null;

          await nextTick();
          drawLossChart();
          drawQualityChart();
          if (logAutoScroll.value && logPreRef.value && (resetToLatest || tailChanged)) {
            logPreRef.value.scrollTop = logPreRef.value.scrollHeight;
          }
        } catch (_) {}
      }

      // Steps at which the task on screen changes its resolution stage (1/4 -> 1/2 -> 1/1); none for a single-stage run.
      // Used by the loss chart and by the quality chart below it, so both mark the same steps.
      function taskStageBoundaries() {
        const tCfg = (activeTask.value && activeTask.value.config && activeTask.value.config.train) || {};
        const stAdv =
          (activeTask.value &&
            activeTask.value.config &&
            activeTask.value.config.advanced &&
            activeTask.value.config.advanced.stage_schedule) ||
          null;
        const totSt = Number((activeTask.value && activeTask.value.total_steps) || tCfg.total_steps || 0);
        if (tCfg.multi_resolution === false || !(totSt >= 9)) return [];
        let s1 = Math.floor((totSt + 2) / 3);
        let s2 = s1 * 2;
        if (stAdv && stAdv.enabled) {
          const rq = Math.max(0, Number(stAdv.r_quarter ?? 15));
          const rh = Math.max(0, Number(stAdv.r_half ?? 25));
          const rf = Math.max(0, Number(stAdv.r_full ?? 60));
          const rSum = Math.max(1e-6, rq + rh + rf);
          s1 = Math.min(totSt, Math.max(0, Math.round((totSt * rq) / rSum)));
          s2 = Math.min(totSt, Math.max(s1, Math.round((totSt * (rq + rh)) / rSum)));
        }
        return [
          { step: s1, label: '1/4 → 1/2' },
          { step: s2, label: '1/2 → 1/1' },
        ];
      }

      // x of a step label that is centred on the grid line at `gx` and still fits the canvas: the label under the last
      // grid line is wider than the right margin, and a long run lost the last digits of its final step there
      function stepLabelX(ctx, label, gx, W) {
        const half = ctx.measureText(label).width / 2;
        return Math.max(half + 2, Math.min(gx, W - 2 - half));
      }

      function drawLossChart() {
        const canvas = lossCanvas.value;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
        canvas.width = Math.max(300, rect.width) * dpr;
        canvas.height = Math.max(200, rect.height) * dpr;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);

        const W = Math.max(300, rect.width);
        const H = Math.max(200, rect.height);
        ctx.clearRect(0, 0, W, H);

        const pts = (telemetry.steps || []).filter((s) => Number.isFinite(s.total) && s.total >= 0);
        if (!pts.length) {
          ctx.fillStyle = '#64748b';
          ctx.font = '12px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(t('monitor.lossWaitingCanvas'), W / 2, H / 2);
          return;
        }

        const padL = 62, padR = 24, padT = 18, padB = 30;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        const xs = pts.map((p) => p.step);
        const rawYs = pts.map((p) => p.total);
        const smYs = [];
        let last = rawYs[0];
        const w = Number(chartCfg.smooth || 0);
        for (const y of rawYs) {
          last = last * w + y * (1 - w);
          smYs.push(last);
        }

        const tf = (v) => (chartCfg.logScale ? Math.log10(Math.max(1e-10, v)) : v);
        const tRaw = rawYs.map(tf);
        const tSm = smYs.map(tf);
        const vPts = (telemetry.vals || []).filter((v) => Number.isFinite(v.mse) && v.mse >= 0);
        const tVal = vPts.map((v) => tf(v.mse));

        const allY = [...tRaw, ...tSm, ...tVal];
        let minX = Math.min(...xs), maxX = Math.max(...xs);
        if (minX === maxX) maxX = minX + 1;
        let minY = Math.min(...allY), maxY = Math.max(...allY);
        if (minY === maxY) { minY -= 0.5; maxY += 0.5; }

        const toX = (x) => padL + ((x - minX) / (maxX - minX)) * plotW;
        const toY = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#64748b';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
          const gy = padT + (i / 5) * plotH;
          const val = maxY - (i / 5) * (maxY - minY);
          const disp = chartCfg.logScale ? Math.pow(10, val).toExponential(1) : val.toFixed(5);
          ctx.beginPath();
          ctx.moveTo(padL, gy);
          ctx.lineTo(W - padR, gy);
          ctx.stroke();
          ctx.fillText(disp, padL - 8, gy + 4);
        }

        ctx.textAlign = 'center';
        const xTicks = 5;
        for (let i = 0; i <= xTicks; i++) {
          const stVal = Math.round(minX + (i / xTicks) * (maxX - minX));
          const gx = padL + (i / xTicks) * plotW;
          ctx.strokeStyle = 'rgba(30, 41, 59, 0.55)';
          ctx.beginPath();
          ctx.moveTo(gx, padT);
          ctx.lineTo(gx, padT + plotH);
          ctx.stroke();
          ctx.fillStyle = '#64748b';
          ctx.fillText(`Step ${stVal}`, stepLabelX(ctx, `Step ${stVal}`, gx, W), H - 8);
        }

        // Multi-Resolution Stage Boundary Markers (1/4 -> 1/2 -> 1/1)
        const stageBounds = taskStageBoundaries();
        if (stageBounds.length) {
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
          ctx.fillStyle = 'rgba(251, 191, 36, 0.85)';
          ctx.font = '10px monospace';
          ctx.textAlign = 'left';
          stageBounds.forEach((b) => {
            if (b.step > minX && b.step < maxX) {
              const bx = toX(b.step);
              ctx.beginPath();
              ctx.moveTo(bx, padT);
              ctx.lineTo(bx, padT + plotH);
              ctx.stroke();
              ctx.fillText(b.label, bx + 4, padT + 12);
            }
          });
          ctx.restore();
        }

        // Raw Loss Line (faint)
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.28)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        pts.forEach((p, i) => {
          const cx = toX(p.step), cy = toY(tRaw[i]);
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        });
        ctx.stroke();

        // Smoothed Loss Line
        ctx.strokeStyle = '#818cf8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach((p, i) => {
          const cx = toX(p.step), cy = toY(tSm[i]);
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        });
        ctx.stroke();

        // Val MSE Points
        if (vPts.length) {
          ctx.strokeStyle = '#34d399';
          ctx.fillStyle = '#10b981';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          vPts.forEach((v, i) => {
            const cx = toX(v.step), cy = toY(tVal[i]);
            if (i === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
          });
          ctx.stroke();
          vPts.forEach((v, i) => {
            const cx = toX(v.step), cy = toY(tVal[i]);
            ctx.beginPath();
            ctx.arc(cx, cy, 3, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }

      // ───────── Training quality chart (three plots over the step axis of the loss chart) ─────────
      // Step range of the quality chart = the step range of the loss chart (the logged loss steps), widened only if a
      // measurement lies outside of it, so the two charts line up.
      function qualityStepRange(points) {
        let lo = Infinity;
        let hi = -Infinity;
        for (const s of telemetry.steps || []) {
          if (!Number.isFinite(s.total) || s.total < 0 || !Number.isFinite(s.step)) continue;
          if (s.step < lo) lo = s.step;
          if (s.step > hi) hi = s.step;
        }
        for (const p of points) {
          if (p.step < lo) lo = p.step;
          if (p.step > hi) hi = p.step;
        }
        if (!Number.isFinite(lo)) return [0, 1];
        return lo === hi ? [lo, lo + 1] : [lo, hi];
      }

      let _qualityGeo = null; // x position of every measurement in the last drawing: pointer -> nearest measurement

      // The chart is drawn for the size its canvas has on the page. Redraw whenever that size changes - the window, the
      // side panel, or styles that arrive a moment after the card first appears - instead of waiting for the next poll
      // with a stretched picture.
      let _qualityWatched = null;
      let _qualityWatcher = null;
      function watchQualityCanvas(canvas) {
        if (_qualityWatched === canvas || typeof ResizeObserver === 'undefined') return;
        if (_qualityWatcher) _qualityWatcher.disconnect();
        _qualityWatched = canvas;
        const first = canvas.getBoundingClientRect();
        let seenW = first.width;
        let seenH = first.height;
        _qualityWatcher = new ResizeObserver((entries) => {
          const box = entries[entries.length - 1].contentRect;
          if (Math.abs(box.width - seenW) < 0.5 && Math.abs(box.height - seenH) < 0.5) return;
          seenW = box.width;
          seenH = box.height;
          drawQualityChart();
        });
        _qualityWatcher.observe(canvas);
      }

      function drawQualityChart() {
        const canvas = qualityCanvas.value;
        const view = qualityView.value;
        _qualityGeo = null;
        if (!canvas || !view || view.state !== 'ready') return;
        watchQualityCanvas(canvas);
        const rect = canvas.getBoundingClientRect();
        const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
        const W = Math.max(300, rect.width);
        const H = Math.max(260, rect.height);
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, W, H);

        const SURFACE = '#020617', GRID = '#1e293b', REF = '#334155', MUTED = '#64748b', TITLE = '#94a3b8', INK = '#e2e8f0';
        const pts = view.points;
        const padL = 62, padR = 24; // the margins of the loss chart: the step axes of the two charts line up
        const plotW = W - padL - padR;
        const right = padL + plotW;
        const [minX, maxX] = qualityStepRange(pts);
        const toX = (s) => padL + ((s - minX) / (maxX - minX)) * plotW;
        const xs = pts.map((p) => toX(p.step));

        // three plots stacked over one step axis; every plot has one scale of its own (no second axis anywhere)
        const titleH = 20, gapH = 8, axisH = 24, topPad = 2;
        const free = H - topPad - axisH - 3 * titleH - 2 * gapH;
        const hFit = Math.max(36, Math.round(free * 0.2));
        const hMain = Math.max(48, Math.floor((free - hFit) / 2));
        const plots = [];
        let yCur = topPad;
        for (const h of [hMain, hMain, hFit]) {
          yCur += titleH;
          plots.push({ top: yCur, bottom: yCur + h, h });
          yCur += h + gapH;
        }

        // keyframe fit in dB: its own range around the values (a dB scale has no meaningful zero)
        const fitVals = pts.map((p) => p.kf_fit_roi_db).filter((v) => v !== null);
        let fitLo = 0;
        let fitHi = 1;
        if (fitVals.length) {
          fitLo = Math.floor(Math.min(...fitVals) - 0.5);
          fitHi = Math.ceil(Math.max(...fitVals) + 0.5);
          if (fitHi - fitLo < 2) {
            fitLo -= 1;
            fitHi += 1;
          }
        }
        const unitY = (p) => (v) => p.bottom - (Math.min(QUALITY_AXIS_MAX, Math.max(0, v)) / QUALITY_AXIS_MAX) * p.h;
        const yOf = [unitY(plots[0]), unitY(plots[1]), (v) => plots[2].bottom - ((v - fitLo) / (fitHi - fitLo)) * plots[2].h];

        // horizontal grid and value labels
        plots.forEach((p, k) => {
          const ticks = k < 2 ? [0, 0.5, 1] : fitVals.length ? (fitHi - fitLo >= 4 ? [fitLo, (fitLo + fitHi) / 2, fitHi] : [fitLo, fitHi]) : [];
          ctx.font = '11px monospace';
          ctx.textAlign = 'right';
          ctx.lineWidth = 1;
          for (const v of ticks) {
            const gy = yOf[k](v);
            ctx.strokeStyle = k < 2 && (v === 0 || (k === 0 && v === 1)) ? REF : GRID;
            ctx.beginPath();
            ctx.moveTo(padL, gy);
            ctx.lineTo(right, gy);
            ctx.stroke();
            ctx.fillStyle = MUTED;
            ctx.fillText(k < 2 ? qualityPct(v) : (Number.isInteger(v) ? String(v) : v.toFixed(1)) + ' dB', padL - 8, gy + 4);
          }
          if (k === 2 && !fitVals.length) {
            ctx.fillStyle = MUTED;
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(t('monitor.qNoValue'), padL + plotW / 2, (p.top + p.bottom) / 2 + 4);
          }
        });

        // step grid and step labels: the same six positions as in the loss chart
        ctx.textAlign = 'center';
        ctx.font = '11px monospace';
        for (let i = 0; i <= 5; i++) {
          const gx = padL + (i / 5) * plotW;
          ctx.strokeStyle = 'rgba(30, 41, 59, 0.55)';
          for (const p of plots) {
            ctx.beginPath();
            ctx.moveTo(gx, p.top);
            ctx.lineTo(gx, p.bottom);
            ctx.stroke();
          }
          ctx.fillStyle = MUTED;
          const stepLabel = `Step ${Math.round(minX + (i / 5) * (maxX - minX))}`;
          ctx.fillText(stepLabel, stepLabelX(ctx, stepLabel, gx, W), H - 8);
        }

        // resolution-stage boundaries, as in the loss chart
        const stageBounds = taskStageBoundaries();
        if (stageBounds.length) {
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
          ctx.fillStyle = 'rgba(251, 191, 36, 0.85)';
          ctx.font = '10px monospace';
          ctx.textAlign = 'left';
          stageBounds.forEach((b) => {
            if (b.step > minX && b.step < maxX) {
              const bx = toX(b.step);
              for (const p of plots) {
                ctx.beginPath();
                ctx.moveTo(bx, p.top);
                ctx.lineTo(bx, p.bottom);
                ctx.stroke();
              }
              ctx.fillText(b.label, bx + 4, plots[0].top + 12);
            }
          });
          ctx.restore();
        }

        // the plate's own texture drift: what a perfect result would still show (named in the key of that plot)
        const floorShown = view.floor1 !== null && view.floor1 >= 0 && view.floor1 <= QUALITY_AXIS_MAX;
        const FLOOR_STROKE = 'rgba(201, 133, 0, 0.7)';
        if (floorShown) {
          const fy = yOf[1](view.floor1);
          ctx.save();
          ctx.setLineDash([2, 3]);
          ctx.strokeStyle = FLOOR_STROKE;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(padL, fy);
          ctx.lineTo(right, fy);
          ctx.stroke();
          ctx.restore();
        }

        // A curve's marker: its shape tells the two curves of a plot apart without relying on colour. Drawn with a
        // ring in the surface colour so it stays readable where it sits on a line.
        const marker = (x, y, s, ring) => {
          if (s.marker === 'square') {
            if (ring) {
              ctx.fillStyle = SURFACE;
              ctx.fillRect(x - 6, y - 6, 12, 12);
            }
            ctx.fillStyle = s.color;
            ctx.fillRect(x - 4, y - 4, 8, 8);
            return;
          }
          if (ring) {
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fillStyle = SURFACE;
            ctx.fill();
          }
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = s.color;
          ctx.fill();
        };

        // curves: a 2 px line through neighbouring measurements (a gap where a value is missing) and a marker at the end
        const keys = [[], [], []]; // per plot: what its key line shows (name + latest value of every curve)
        for (const s of QUALITY_SERIES) {
          const p = plots[s.plot];
          const toY = yOf[s.plot];
          const vals = pts.map((q) => q[s.field]);
          ctx.strokeStyle = s.color;
          ctx.lineWidth = 2;
          ctx.lineJoin = 'round';
          ctx.lineCap = 'round';
          ctx.beginPath();
          let open = false;
          let last = -1;
          vals.forEach((v, i) => {
            if (v === null) {
              open = false;
              return;
            }
            if (open) ctx.lineTo(xs[i], toY(v));
            else ctx.moveTo(xs[i], toY(v));
            open = true;
            last = i;
          });
          ctx.stroke();
          ctx.fillStyle = s.color;
          vals.forEach((v, i) => {
            if (v === null) return;
            const lone = (i === 0 || vals[i - 1] === null) && (i === vals.length - 1 || vals[i + 1] === null);
            if (lone && i !== last) {
              // a measurement with no neighbour to connect to
              ctx.beginPath();
              ctx.arc(xs[i], toY(v), 3, 0, Math.PI * 2);
              ctx.fill();
            }
            if (s.plot < 2 && v > QUALITY_AXIS_MAX) {
              // above the fixed axis: drawn at the top edge and marked
              ctx.beginPath();
              ctx.moveTo(xs[i], p.top + 1);
              ctx.lineTo(xs[i] - 4, p.top + 8);
              ctx.lineTo(xs[i] + 4, p.top + 8);
              ctx.closePath();
              ctx.fill();
            }
          });
          if (last >= 0) marker(xs[last], toY(vals[last]), s, true);
          keys[s.plot].push({
            series: s,
            text: `${t('monitor.qShort_' + s.id)} ${last >= 0 ? qualitySeriesText(s, vals[last]) : '—'}`,
          });
        }
        if (floorShown) keys[1].push({ series: null, text: t('monitor.qTileSub_floor', { v: qualityPct(view.floor1) }) });

        // The line above every plot: its title on the left, on the right the key of its curves with their latest values.
        // All text of the chart lives in these lines and on the axes: nothing is written over the curves.
        const titles = [t('monitor.qPlot_arrive'), t('monitor.qPlot_drift'), t('monitor.qPlot_fit')];
        plots.forEach((p, k) => {
          const base = p.top - 8; // clear of a marker that sits on the top edge of the plot (a value above the axis)
          ctx.font = '11px sans-serif';
          const keyW = 18;
          const pad = 5;
          const gap = 14;
          const widths = keys[k].map((e) => keyW + pad + ctx.measureText(e.text).width);
          let x = right - widths.reduce((a, b) => a + b, 0) - gap * Math.max(0, keys[k].length - 1);
          const keysLeft = x;
          ctx.textAlign = 'left';
          keys[k].forEach((e, i) => {
            const mid = base - 4;
            if (e.series) {
              ctx.strokeStyle = e.series.color;
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(x, mid);
              ctx.lineTo(x + keyW, mid);
              ctx.stroke();
              marker(x + keyW / 2, mid, e.series, false);
            } else {
              ctx.save();
              ctx.setLineDash([2, 3]);
              ctx.strokeStyle = FLOOR_STROKE;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(x, mid);
              ctx.lineTo(x + keyW, mid);
              ctx.stroke();
              ctx.restore();
            }
            ctx.fillStyle = INK;
            ctx.fillText(e.text, x + keyW + pad, base);
            x += widths[i] + gap;
          });
          // the title takes what is left of the line (cut with an ellipsis on a narrow chart)
          let title = titles[k];
          const room = keysLeft - 16 - padL;
          if (ctx.measureText(title).width > room) {
            while (title.length > 1 && ctx.measureText(title + '…').width > room) title = title.slice(0, -1);
            title = title.length > 1 ? title + '…' : '';
          }
          ctx.fillStyle = TITLE;
          ctx.fillText(title, padL, base);
        });

        // crosshair on the measurement under the pointer / reached with the arrow keys
        const hi = qualityUi.hoverPoint;
        if (hi >= 0 && hi < pts.length) {
          const hx = xs[hi];
          ctx.strokeStyle = MUTED;
          ctx.lineWidth = 1;
          for (const p of plots) {
            ctx.beginPath();
            ctx.moveTo(hx, p.top);
            ctx.lineTo(hx, p.bottom);
            ctx.stroke();
          }
          for (const s of QUALITY_SERIES) {
            const v = pts[hi][s.field];
            if (v !== null) marker(hx, yOf[s.plot](v), s, true);
          }
          qualityUi.hoverLeft = (hx / W) * 100;
        }
        _qualityGeo = { xs };
      }

      function setQualityHoverPoint(i) {
        const v = qualityView.value;
        const n = v && v.state === 'ready' ? v.points.length : 0;
        const next = i >= 0 && i < n ? i : -1;
        if (next === qualityUi.hoverPoint) return;
        qualityUi.hoverPoint = next;
        drawQualityChart();
      }

      // the pointer does not have to hit a curve: the nearest measurement along the step axis is taken
      function onQualityChartMove(e) {
        const geo = _qualityGeo;
        const canvas = qualityCanvas.value;
        if (!geo || !canvas || !geo.xs.length) return;
        const x = e.clientX - canvas.getBoundingClientRect().left;
        let best = 0;
        let bestDist = Infinity;
        geo.xs.forEach((px, i) => {
          const dist = Math.abs(px - x);
          if (dist < bestDist) {
            bestDist = dist;
            best = i;
          }
        });
        setQualityHoverPoint(best);
      }

      function clearQualityHover() {
        setQualityHoverPoint(-1);
      }

      // arrow keys on the focused chart: from one measurement to the next (starts at the latest one)
      function stepQualityHover(delta) {
        const v = qualityView.value;
        const n = v && v.state === 'ready' ? v.points.length : 0;
        if (!n) return;
        const cur = qualityUi.hoverPoint;
        setQualityHoverPoint(cur < 0 ? n - 1 : Math.min(n - 1, Math.max(0, cur + delta)));
      }

      // the same for the per-frame strip: from one evaluated frame to the next (starts at the weakest frame)
      function stepQualityFrame(delta) {
        const v = qualityView.value;
        const strip = v && v.state === 'ready' ? v.strip : null;
        if (!strip || !strip.bars.length) return;
        const cur = qualityUi.hoverFrame;
        const start = strip.weakest ? strip.bars.indexOf(strip.weakest) : 0;
        qualityUi.hoverFrame = cur < 0 ? Math.max(0, start) : Math.min(strip.bars.length - 1, Math.max(0, cur + delta));
      }

      // ───────── Full-Frame Viewer Controls ─────────
      function stepFrame(delta) {
        if (!viewer.frames.length) return;
        viewer.index = (viewer.index + delta + viewer.frames.length) % viewer.frames.length;
      }

      function togglePlay() {
        viewer.playing = !viewer.playing;
      }

      function resetViewerZoom() {
        viewer.zoom = 1;
        viewer.panX = 0;
        viewer.panY = 0;
      }

      function onViewerWheel(e) {
        const factor = e.deltaY < 0 ? 1.14 : 0.88;
        viewer.zoom = Math.min(12, Math.max(0.25, viewer.zoom * factor));
      }

      function onViewerMouseDown(e) {
        viewer.dragging = true;
        viewer.dragStartX = e.clientX - viewer.panX;
        viewer.dragStartY = e.clientY - viewer.panY;
      }

      function onViewerMouseMove(e) {
        if (!viewer.dragging) return;
        viewer.panX = e.clientX - viewer.dragStartX;
        viewer.panY = e.clientY - viewer.dragStartY;
      }

      function onViewerMouseUp() {
        viewer.dragging = false;
      }

      // ───────── Nuke 3x3 Contact Sheet Viewer Controls ─────────
      function stepContactFrame(delta) {
        if (!contactViewer.frames.length) return;
        contactViewer.index = (contactViewer.index + delta + contactViewer.frames.length) % contactViewer.frames.length;
      }

      function toggleContactPlay() {
        contactViewer.playing = !contactViewer.playing;
      }

      function resetContactZoom() {
        contactViewer.zoom = 1;
        contactViewer.panX = 0;
        contactViewer.panY = 0;
      }

      function onContactWheel(e) {
        const factor = e.deltaY < 0 ? 1.14 : 0.88;
        contactViewer.zoom = Math.min(16, Math.max(0.4, contactViewer.zoom * factor));
      }

      function onContactMouseDown(e) {
        contactViewer.dragging = true;
        contactViewer.dragStartX = e.clientX - contactViewer.panX;
        contactViewer.dragStartY = e.clientY - contactViewer.panY;
      }

      function onContactMouseMove(e) {
        if (!contactViewer.dragging) return;
        contactViewer.panX = e.clientX - contactViewer.dragStartX;
        contactViewer.panY = e.clientY - contactViewer.dragStartY;
      }

      function onContactMouseUp() {
        contactViewer.dragging = false;
      }

      // ───────── Task, Model Downloads & Log Operations ─────────
      async function stopTask(taskId) {
        try {
          const res = await apiFetch(`/api/v1/tasks/${taskId}/stop`, { method: 'POST' });
          if (res.task && res.task.status === 'cancelled') {
            notify(t('msg.queueCancelledToast'), 'warning');
          } else {
            notify(t('msg.taskPausedSavedToast'), 'warning');
          }
          await fetchQueueAndTasks();
          await fetchCheckpoints();
          if (topNav.value === 'studio' && studioMode.value === 'monitor') await fetchTelemetry(false);
          if (topNav.value === 'admin') await loadAdminData();
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function deleteTask(taskObj) {
        if (!confirm(t('msg.confirmDeleteTask', { node: taskObj.node_name }))) return;
        try {
          await apiFetch(`/api/v1/tasks/${taskObj.id}`, { method: 'DELETE' });
          notify(t('msg.taskDeletedToast', { node: taskObj.node_name }), 'warning');
          await fetchQueueAndTasks();
          await fetchCheckpoints();
          if (!tasks.value.length) {
            studioMode.value = 'create';
          }
          if (topNav.value === 'admin') await loadAdminData();
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      function triggerBrowserDownload(url, filename = '') {
        const a = document.createElement('a');
        a.href = url;
        if (filename) a.download = filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          if (a.parentNode) a.parentNode.removeChild(a);
        }, 1500);
      }

      async function downloadModel(taskId, ext = 'pt', step = null) {
        try {
          const item = step != null
            ? taskModels.items.find((x) => x.step === step)
            : (taskModels.items[0] || null);
          let alreadyReady = false;
          let dlFilename = '';
          if (item) {
            {
              alreadyReady = !!item.has_pt;
              dlFilename = item.pt_filename || '';
            }
          }
          if (!alreadyReady) {
            const stLabel = step != null ? `Step ${step}` : t('monitor.tagLatest');
            notify(t('msg.preparingModelToast', { label: stLabel }), 'info');
            const q = step != null ? `?step=${encodeURIComponent(step)}` : '';
            const expRes = await apiFetch(`/api/v1/tasks/${taskId}/export-cat${q}`, { method: 'POST' });
            if (ext === 'pt' && expRes.pt_filename) dlFilename = expRes.pt_filename;
            await fetchTelemetry(false);
          }
          const stepParam = step != null ? `&step=${encodeURIComponent(step)}` : '';
          const langParam = currentLang.value ? `&lang=${encodeURIComponent(currentLang.value)}` : '';
          const url = `/api/v1/tasks/${taskId}/download/${encodeURIComponent(ext)}?api_key=${encodeURIComponent(apiKey.value)}${stepParam}${langParam}`;
          triggerBrowserDownload(url, dlFilename);
          notify(t('msg.downloadStartedToast', { filename: dlFilename || ('.' + ext) }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function downloadAllModels(taskId, ext = 'pt') {
        if (!taskModels.count && !(activeTask.value && activeTask.value.has_ckpt)) {
          notify(t('msg.noModelsToPackToast'), 'warning');
          return;
        }
        try {
          const zipName = taskModels.zip_filename || `${taskModels.nuke_prefix || 'Training'}.zip`;
          notify(t('msg.packingAllModelsToast', { zip: zipName }), 'info');
          const langParam = currentLang.value ? `&lang=${encodeURIComponent(currentLang.value)}` : '';
          const url = `/api/v1/tasks/${taskId}/download-all/${encodeURIComponent(ext)}?api_key=${encodeURIComponent(apiKey.value)}${langParam}`;
          triggerBrowserDownload(url, zipName);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function copyTrainLog() {
        const errBlock = activeTask.value && activeTask.value.error_msg ? activeTask.value.error_msg + '\n\n' : '';
        const logBlock = (telemetry.tail && telemetry.tail.length) ? telemetry.tail.join('\n') : '';
        const text = (errBlock + logBlock).trim();
        if (!text) {
          notify(t('msg.noLogsToCopyToast'), 'warning');
          return;
        }
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
          } else {
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
          }
          notify(t('msg.copiedLogsToast'), 'success');
        } catch (_) {
          notify(t('msg.copyFailedManual'), 'warning');
        }
      }

      // ───────── Dataset Scrubber, Validation Sequence Attachment & Upload Actions ─────────
      function getAssetScrubber(a) {
        if (!a || !a.id) {
          return {
            loading: false,
            loaded: false,
            mode: 'pair',
            pairIndex: 0,
            seqIndex: 0,
            paired_frames: [],
            seq_frames: [],
          };
        }
        if (!assetFramesCache[a.id]) {
          assetFramesCache[a.id] = {
            loading: false,
            loaded: false,
            mode: a.kind === 'eval_sequence' ? 'seq' : 'pair',
            pairIndex: 0,
            seqIndex: 0,
            paired_frames: [],
            seq_frames: [],
          };
        }
        return assetFramesCache[a.id];
      }

      async function ensureAssetFrames(assetId, force = false) {
        if (!assetId || !user.value) return;
        const st = getAssetScrubber({ id: assetId });
        if ((st.loaded && !force) || st.loading) return;
        st.loading = true;
        try {
          const d = await apiFetch(`/api/v1/assets/${assetId}/frames`);
          st.paired_frames = d.paired_frames || [];
          st.seq_frames = d.seq_frames || [];
          if (st.pairIndex >= st.paired_frames.length) st.pairIndex = 0;
          if (st.seqIndex >= st.seq_frames.length) st.seqIndex = 0;
          if (st.mode === 'seq' && !st.seq_frames.length && st.paired_frames.length) {
            st.mode = 'pair';
          }
          st.loaded = true;
        } catch (_) {
        } finally {
          st.loading = false;
        }
      }

      function setAssetScrubMode(a, mode) {
        if (!a || !a.id) return;
        if (mode === 'seq' && !a.has_eval_sequence && a.kind !== 'eval_sequence') return;
        const st = getAssetScrubber(a);
        st.mode = mode;
        ensureAssetFrames(a.id);
      }

      function stepAssetFrame(a, delta) {
        if (!a || !a.id) return;
        const st = getAssetScrubber(a);
        ensureAssetFrames(a.id);
        if (st.mode === 'seq' || a.kind === 'eval_sequence') {
          const tot = st.seq_frames.length || Number(a.seq_frame_count || a.frame_count || 1);
          if (tot > 0) st.seqIndex = (Number(st.seqIndex || 0) + delta + tot) % tot;
        } else {
          const tot = st.paired_frames.length || Number(a.frame_count || 1);
          if (tot > 0) st.pairIndex = (Number(st.pairIndex || 0) + delta + tot) % tot;
        }
      }

      function currentAssetPairFrame(a) {
        if (!a) return { idx: 0, total: 0, name: '', src_url: '', tgt_url: '' };
        const st = getAssetScrubber(a);
        const list = st.paired_frames || [];
        if (list.length) {
          const idx = Math.min(Math.max(0, Number(st.pairIndex) || 0), list.length - 1);
          const f = list[idx];
          return {
            idx,
            total: list.length,
            name: f.name || '',
            src_url: f.src_url || a.source_thumb_url || '',
            tgt_url: f.tgt_url || a.target_thumb_url || '',
          };
        }
        return {
          idx: 0,
          total: Math.max(0, Number(a.frame_count || 0)),
          name: '',
          src_url: a.source_thumb_url || '',
          tgt_url: a.target_thumb_url || a.thumb_url || '',
        };
      }

      function currentAssetSeqFrame(a) {
        if (!a) return { idx: 0, total: 0, name: '', url: '' };
        const st = getAssetScrubber(a);
        const list = st.seq_frames || [];
        if (list.length) {
          const idx = Math.min(Math.max(0, Number(st.seqIndex) || 0), list.length - 1);
          const f = list[idx];
          return {
            idx,
            total: list.length,
            name: f.name || '',
            url: f.url || a.seq_thumb_url || '',
          };
        }
        const tot = a.kind === 'eval_sequence' ? Number(a.frame_count || 0) : Number(a.seq_frame_count || 0);
        return {
          idx: 0,
          total: Math.max(0, tot),
          name: '',
          url: a.seq_thumb_url || a.source_thumb_url || '',
        };
      }

      function demoBadgeTitle(a) {
        if (!a || !a.demo_key) return '';
        const v = lookupLocalePath(activeMessages.value, `priv.demoTitle_${a.demo_key}`);
        return v ? String(v) : '';
      }

      function demoDescText(a) {
        if (!a || !a.demo_key) return '';
        const v = lookupLocalePath(activeMessages.value, `priv.demoDesc_${a.demo_key}`);
        return v ? String(v) : '';
      }

      function formatDatasetOptionLabel(a) {
        if (!a) return '';
        const seqPart = a.has_eval_sequence
          ? t('wizard.datasetOptSeqPart', { seqFrames: a.seq_frame_count })
          : '';
        return t('wizard.datasetOptionFmt', {
          name: a.name,
          frames: a.frame_count,
          w: a.width,
          h: a.height,
          seqPart,
        });
      }

      function formatLabDatasetOptionLabel(a) {
        if (!a) return '';
        if (a.kind === 'eval_sequence') {
          return t('lab.evalDatasetWithSeqOpt', {
            name: a.name,
            seqFrames: a.frame_count,
            w: a.width,
            h: a.height,
          });
        }
        if (a.has_eval_sequence) {
          return t('lab.evalDatasetWithSeqOpt', {
            name: a.name,
            seqFrames: a.seq_frame_count,
            w: a.width,
            h: a.height,
          });
        }
        return t('lab.evalDatasetNoSeqOpt', {
          name: a.name,
          frames: a.frame_count,
        });
      }

      function openDatasetModal() {
        datasetPickerModal.open = false;
        uploadModal.mode = 'create';
        uploadModal.targetAsset = null;
        uploadModal.kind = 'paired_dataset';
        uploadModal.name = `dataset_${Math.floor(Date.now() / 1000) % 10000}`;
        uploadModal.taskType = '';
        uploadModal.inputColorspace = 'auto';
        uploadModal.enableCheck = true;
        uploadModal.strategy = 'standard';
        uploadModal.maxErrorPct = 35;
        uploadModal.sourceFiles = [];
        uploadModal.targetFiles = [];
        uploadModal.seqFiles = [];
        uploadModal.errorMsg = '';
        uploadModal.open = true;
      }

      const selectedLabEvalExplicitAsset = computed(() => {
        if (!labEvalAssetId.value) return null;
        return evalSequenceAssets.value.find((a) => a.id === labEvalAssetId.value) || null;
      });

      const pickerBaseDatasets = computed(() => {
        return datasetPickerModal.mode === 'lab_eval' ? evalSequenceAssets.value : pairedAssets.value;
      });

      const datasetPickerTitle = computed(() => {
        return datasetPickerModal.mode === 'lab_eval' ? t('datasets.pickerEvalTitle') : t('datasets.pickerTrainTitle');
      });

      const pickerWithSeqCount = computed(() => {
        return pickerBaseDatasets.value.filter((a) => a.kind === 'eval_sequence' || a.has_eval_sequence).length;
      });

      const filteredPickerDatasets = computed(() => {
        const q = (datasetPickerModal.query || '').trim().toLowerCase();
        const sf = datasetPickerModal.seqFilter || 'all';
        const tf = datasetPickerModal.taskTypeFilter || 'all';
        return pickerBaseDatasets.value.filter((a) => {
          if (sf === 'with_seq' && !(a.kind === 'eval_sequence' || a.has_eval_sequence)) {
            return false;
          }
          if (tf !== 'all' && (a.task_type || '') !== tf) {
            return false;
          }
          if (q) {
            const ttLbl = a.task_type ? t('adv.preset_' + a.task_type) : '';
            const hay = `${a.name || ''} ${demoBadgeTitle(a) || ''} ${ttLbl} ${a.width}x${a.height} ${a.in_channels}->${a.out_channels}`.toLowerCase();
            if (!hay.includes(q)) return false;
          }
          return true;
        });
      });

      function openDatasetPicker(mode = 'train', toolId = '') {
        datasetPickerModal.mode = mode === 'lab_eval' ? 'lab_eval' : 'train';
        datasetPickerModal.query = '';
        datasetPickerModal.seqFilter = 'all';
        datasetPickerModal.taskTypeFilter = 'all';
        datasetPickerModal.open = true;
      }

      function isPickerAssetSelected(a) {
        if (datasetPickerModal.mode === 'lab_eval') {
          if (!a) return labEvalAssetId.value === null;
          return labEvalAssetId.value === a.id;
        }
        return !!(selectedDataset.value && a && selectedDataset.value.id === a.id);
      }

      function pickDatasetItem(assetOrNull) {
        if (datasetPickerModal.mode === 'lab_eval') {
          labEvalAssetId.value = assetOrNull ? assetOrNull.id : null;
          datasetPickerModal.open = false;
          return;
        }
        if (!assetOrNull) return;
        selectedDatasetId.value = assetOrNull.id;
        onSelectDataset();
        datasetPickerModal.open = false;
      }

      function onUploadTaskTypeChange() {
        const tt = uploadModal.taskType || '';
        if (tt === 'roto_matte' || tt === 'keying_despill') {
          uploadModal.strategy = 'alpha_matting';
        } else if (uploadModal.strategy === 'alpha_matting') {
          uploadModal.strategy = 'standard';
          uploadModal.maxErrorPct = 35;
        }
      }

      function openAttachSeqModal(assetObj) {
        datasetPickerModal.open = false;
        if (!assetObj || assetObj.kind === 'eval_sequence') {
          openDatasetModal();
          return;
        }
        uploadModal.mode = 'attach_seq';
        uploadModal.targetAsset = assetObj;
        uploadModal.kind = 'paired_dataset';
        uploadModal.name = assetObj.name;
        uploadModal.inputColorspace = 'auto';
        uploadModal.sourceFiles = [];
        uploadModal.targetFiles = [];
        uploadModal.seqFiles = [];
        uploadModal.errorMsg = '';
        uploadModal.open = true;
      }

      function onUploadStrategyChange() {
        if (uploadModal.strategy === 'strict') uploadModal.maxErrorPct = 15;
        else if (uploadModal.strategy === 'standard') uploadModal.maxErrorPct = 35;
        else if (uploadModal.strategy === 'loose') uploadModal.maxErrorPct = 50;
      }

      function onPairFilesSelected(role, e) {
        uploadModal.errorMsg = '';
        const list = Array.from(e.target.files || []);
        if (role === 'source') uploadModal.sourceFiles = list;
        else if (role === 'target') uploadModal.targetFiles = list;
        else if (role === 'seq') uploadModal.seqFiles = list;
      }

      async function doUploadPairedDataset() {
        uploadModal.errorMsg = '';
        if (uploadModal.mode === 'attach_seq' && uploadModal.targetAsset) {
          if (!uploadModal.seqFiles.length) {
            uploadModal.errorMsg = t('msg.errSelectSourceSeq');
            return;
          }
          uploadModal.uploading = true;
          try {
            const aid = uploadModal.targetAsset.id;
            const fd = new FormData();
            fd.append('input_colorspace', uploadModal.inputColorspace || 'auto');
            fd.append('working_colorspace', 'ACEScg');
            for (const f of uploadModal.seqFiles) {
              fd.append('seq_files', f);
            }
            const res = await apiFetch(`/api/v1/assets/${aid}/eval-sequence`, {
              method: 'POST',
              body: fd,
            });
            delete assetFramesCache[aid];
            await fetchAssets();
            ensureAssetFrames(aid, true);
            if (topNav.value === 'lab' && labMode.value === 'analyze') {
              await fetchLabCompare(true);
            }
            notify(
              t('msg.evalSeqAttachedToast', {
                name: res.asset.name,
                frames: res.asset.seq_frame_count,
              }),
              'success'
            );
            uploadModal.open = false;
          } catch (e) {
            uploadModal.errorMsg = e.message;
          } finally {
            uploadModal.uploading = false;
          }
          return;
        }

        if (!uploadModal.sourceFiles.length || !uploadModal.targetFiles.length) {
          uploadModal.errorMsg = t('msg.errSelectBothSeq');
          return;
        }
        uploadModal.uploading = true;
        try {
          const fd = new FormData();
          fd.append('name', uploadModal.name || 'paired_dataset');
          fd.append('kind', 'paired_dataset');
          fd.append('input_colorspace', uploadModal.inputColorspace || 'auto');
          fd.append('working_colorspace', 'ACEScg');
          fd.append('enable_check', uploadModal.enableCheck ? 'true' : 'false');
          fd.append('strategy', uploadModal.strategy || 'standard');
          fd.append('task_type', uploadModal.taskType || '');
          fd.append('max_error_pct', String(uploadModal.maxErrorPct || 35));
          for (const f of uploadModal.sourceFiles) {
            fd.append('source_files', f);
          }
          for (const f of uploadModal.targetFiles) {
            fd.append('target_files', f);
          }
          for (const f of uploadModal.seqFiles) {
            fd.append('seq_files', f);
          }
          const res = await apiFetch('/api/v1/assets/upload-pair', {
            method: 'POST',
            body: fd,
          });
          delete assetFramesCache[res.asset.id];
          await fetchAssets();
          selectedDatasetId.value = res.asset.id;
          syncFormWithSelectedDataset();
          ensureAssetFrames(res.asset.id, true);
          if (res.asset.seq_frame_count >= 2) {
            notify(
              t('msg.datasetCreatedWithSeqToast', {
                name: res.asset.name,
                frames: res.asset.frame_count,
                seqFrames: res.asset.seq_frame_count,
                mae: (res.asset.avg_mae * 100).toFixed(2),
              }),
              'success'
            );
          } else {
            notify(
              t('msg.datasetCreatedToast', {
                name: res.asset.name,
                frames: res.asset.frame_count,
                mae: (res.asset.avg_mae * 100).toFixed(2),
              }),
              'success'
            );
          }
          uploadModal.open = false;
        } catch (e) {
          uploadModal.errorMsg = e.message;
        } finally {
          uploadModal.uploading = false;
        }
      }

      async function detachEvalSequence(a) {
        if (!a || !a.id) return;
        if (!confirm(t('msg.confirmRemoveEvalSeq', { name: a.name }))) return;
        try {
          await apiFetch(`/api/v1/assets/${a.id}/eval-sequence`, { method: 'DELETE' });
          delete assetFramesCache[a.id];
          await fetchAssets();
          ensureAssetFrames(a.id, true);
          if (topNav.value === 'lab' && labMode.value === 'analyze') {
            await fetchLabCompare(true);
          }
          notify(t('msg.evalSeqRemovedToast', { name: a.name }), 'info');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function updateDatasetTaskType(assetObj, newTaskType) {
        if (!assetObj || !assetObj.id) return;
        const clean = String(newTaskType || '').trim();
        try {
          const res = await apiFetch(`/api/v1/assets/${assetObj.id}`, {
            method: 'PATCH',
            body: JSON.stringify({ task_type: clean }),
          });
          if (res && res.asset) {
            const idx = assets.value.findIndex((x) => x.id === assetObj.id);
            if (idx >= 0) Object.assign(assets.value[idx], res.asset);
            const typeLbl = clean ? t('adv.preset_' + clean) : t('upload.taskTypeUnspecified');
            notify(t('msg.datasetTaskTypeUpdatedToast', { name: res.asset.name, type: typeLbl }), 'success');
          }
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function importDemoAssets() {
        try {
          await apiFetch('/api/v1/assets/import-demos', { method: 'POST' });
          await fetchAssets();
          if (topNav.value === 'assets') {
            for (const a of assets.value) {
              ensureAssetFrames(a.id);
            }
          }
          notify(t('msg.demosRestoredToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      function useDatasetForTraining(a) {
        if (a.kind === 'eval_sequence') {
          useAssetForLabEval(a);
          return;
        }
        selectedDatasetId.value = a.id;
        syncFormWithSelectedDataset();
        ensureAssetFrames(a.id);
        topNav.value = 'studio';
        studioMode.value = 'create';
      }

      function useAssetForLabEval(a) {
        labEvalAssetId.value = a.id;
        topNav.value = 'lab';
        labMode.value = 'analyze';
      }

      async function deleteAsset(a) {
        if (!confirm(t('msg.confirmDeleteAsset', { name: a.name }))) return;
        try {
          await apiFetch(`/api/v1/assets/${a.id}`, { method: 'DELETE' });
          delete assetFramesCache[a.id];
          await fetchAssets();
          notify(t('msg.assetDeletedToast', { name: a.name }), 'warning');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      // ───────── API Key & Converter Tools ─────────
      async function copyMyApiKey() {
        if (!canUseApi.value) {
          notify(t('server.tempUserApiLocked'), 'warning');
          return;
        }
        if (user.value && user.value.api_key) {
          await copyTextValue(user.value.api_key, 'msg.apiKeyCopiedToast');
        }
      }

      async function regenerateApiKey() {
        if (!canUseApi.value) {
          notify(t('server.tempUserApiLocked'), 'warning');
          return;
        }
        try {
          const d = await apiFetch('/api/v1/auth/regenerate-key', { method: 'POST' });
          apiKey.value = d.api_key;
          if (user.value) user.value.api_key = d.api_key;
          notify(t('msg.apiKeyRegeneratedToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }


      // ───────── Admin Console Actions ─────────
      async function switchToAdmin() {
        topNav.value = 'admin';
        if (typeof window !== 'undefined' && window.history && window.location.pathname !== '/admin') {
          window.history.pushState({}, '', '/admin');
        }
        if (adminToken.value) {
          await loadAdminData();
        }
      }

      async function goBackToStudio() {
        topNav.value = 'studio';
        if (typeof window !== 'undefined' && window.history && window.location.pathname === '/admin') {
          window.history.pushState({}, '', '/');
        }
        await restoreSession();
      }

      const adminGpuList = computed(() => {
        if (Array.isArray(adminSettings.gpu_pool.gpus) && adminSettings.gpu_pool.gpus.length) {
          return adminSettings.gpu_pool.gpus;
        }
        return Array.isArray(queue.gpus) ? queue.gpus : [];
      });

      const adminEnabledGpuCount = computed(() =>
        adminGpuList.value.filter((g) => g && g.enabled !== false).length
      );

      function _applyAdminSettingsPayload(sRes) {
        if (!sRes) return;
        if (sRes.settings) {
          adminSettings.max_jobs_per_gpu = Number(sRes.settings.max_jobs_per_gpu || 1);
          if (typeof sRes.settings.enable_temp_user_restrictions === 'boolean') {
            adminSettings.enable_temp_user_restrictions = sRes.settings.enable_temp_user_restrictions;
          }
          if (sRes.settings.default_user_max_tasks != null) {
            adminSettings.default_user_max_tasks = Number(sRes.settings.default_user_max_tasks);
          }
          if (sRes.settings.default_user_max_epochs != null) {
            adminSettings.default_user_max_epochs = Number(sRes.settings.default_user_max_epochs);
          }
          if (sRes.settings.max_running_per_user != null) {
            adminSettings.max_running_per_user = Number(sRes.settings.max_running_per_user);
          }
          adminSettings.data_cache_root = String(sRes.settings.data_cache_root || '');
        }
        if (sRes.gpu_pool) {
          adminSettings.gpu_pool.gpus = Array.isArray(sRes.gpu_pool.gpus) ? sRes.gpu_pool.gpus : [];
          adminSettings.gpu_pool.total_gpu_count = Number(
            sRes.gpu_pool.total_gpu_count ?? adminSettings.gpu_pool.gpus.length
          );
          adminSettings.gpu_pool.enabled_gpu_count = Number(
            sRes.gpu_pool.enabled_gpu_count ??
              adminSettings.gpu_pool.gpus.filter((g) => g.enabled !== false).length
          );
          adminSettings.gpu_pool.all_gpus_disabled = !!sRes.gpu_pool.all_gpus_disabled;
          adminSettings.gpu_pool.primary_gpu = sRes.gpu_pool.primary_gpu ?? null;
          adminSettings.gpu_pool.secondary_gpu = sRes.gpu_pool.secondary_gpu ?? null;
          adminSettings.gpu_pool.disabled_gpus = Array.isArray(sRes.gpu_pool.disabled_gpus)
            ? sRes.gpu_pool.disabled_gpus.map(String)
            : [];
        }
        if (sRes.storage) {
          adminSettings.storage.data_cache_root = String(sRes.storage.data_cache_root || '');
          adminSettings.storage.is_custom = !!sRes.storage.is_custom;
          adminSettings.storage.default_root_hint = String(sRes.storage.default_root_hint || '');
          if (sRes.storage.effective_paths) {
            Object.assign(adminSettings.storage.effective_paths, sRes.storage.effective_paths);
          }
          if (sRes.storage.disk) {
            Object.assign(adminSettings.storage.disk, sRes.storage.disk);
          }
        }
        if (sRes.db_summary) {
          Object.assign(adminSettings.db_summary, sRes.db_summary);
        }
      }

      function _normalizeAdminUsers(list) {
        return (list || []).map((u) =>
          Object.assign({}, u, {
            edit_max_tasks: Number(u.max_tasks ?? 2),
            edit_max_epochs: Number(u.raw_max_epochs ?? u.max_epochs ?? 500000),
          })
        );
      }

      async function loadAdminData() {
        if (!adminToken.value) return;
        try {
          const [sRes, uRes, tRes] = await Promise.all([
            apiFetch('/api/v1/admin/settings'),
            apiFetch('/api/v1/admin/users'),
            apiFetch('/api/v1/tasks?all_users=true'),
          ]);
          _applyAdminSettingsPayload(sRes);
          adminUsers.value = _normalizeAdminUsers(uRes.users || []);
          adminTasks.value = tRes.tasks || [];
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function saveAdminSettings() {
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({
              max_jobs_per_gpu: Number(adminSettings.max_jobs_per_gpu || 1),
            }),
          });
          _applyAdminSettingsPayload(sRes);
          await fetchQueueAndTasks(true);
          notify(t('msg.settingsSavedToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function toggleAdminGpuEnabled(gpuObj) {
        if (!gpuObj || adminSettings.savingGpuPool) return;
        adminSettings.savingGpuPool = true;
        const gid = String(gpuObj.gpu_id);
        const curDis = new Set(
          adminGpuList.value.filter((g) => g.enabled === false).map((g) => String(g.gpu_id))
        );
        if (gpuObj.enabled === false) {
          curDis.delete(gid);
        } else {
          curDis.add(gid);
        }
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({
              disabled_gpus: Array.from(curDis),
            }),
          });
          _applyAdminSettingsPayload(sRes);
          await fetchQueueAndTasks(true);
          notify(t('msg.settingsSavedToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          adminSettings.savingGpuPool = false;
        }
      }

      async function enableAllAdminGpus() {
        if (adminSettings.savingGpuPool) return;
        adminSettings.savingGpuPool = true;
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({
              disabled_gpus: [],
            }),
          });
          _applyAdminSettingsPayload(sRes);
          await fetchQueueAndTasks(true);
          notify(t('msg.settingsSavedToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          adminSettings.savingGpuPool = false;
        }
      }

      async function setAdminGpuRole(gpuObj, role) {
        if (!gpuObj || gpuObj.enabled === false || adminSettings.savingGpuPool) return;
        const gid = String(gpuObj.gpu_id);
        if (role === 'primary' && gpuObj.is_primary) return;
        if (role === 'secondary' && (gpuObj.is_secondary || adminEnabledGpuCount.value < 2)) return;
        adminSettings.savingGpuPool = true;
        const payload = role === 'primary' ? { primary_gpu: gid } : { secondary_gpu: gid };
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify(payload),
          });
          _applyAdminSettingsPayload(sRes);
          await fetchQueueAndTasks(true);
          notify(t('msg.settingsSavedToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          adminSettings.savingGpuPool = false;
        }
      }

      async function toggleTempUserRestrictions() {
        if (adminSettings.togglingRestrictions) return;
        adminSettings.togglingRestrictions = true;
        const nextVal = !adminSettings.enable_temp_user_restrictions;
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({
              enable_temp_user_restrictions: nextVal,
            }),
          });
          _applyAdminSettingsPayload(sRes);
          await Promise.all([loadAdminData(), refreshMeQuietly()]);
          notify(
            t(nextVal ? 'msg.tempRestrictEnabledToast' : 'msg.tempRestrictDisabledToast'),
            nextVal ? 'warning' : 'success'
          );
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          adminSettings.togglingRestrictions = false;
        }
      }

      async function saveDefaultTempQuotas(syncExisting = false) {
        if (adminSettings.savingQuotas) return;
        adminSettings.savingQuotas = true;
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({
              default_user_max_tasks: Math.max(0, Math.floor(Number(adminSettings.default_user_max_tasks ?? 2))),
              default_user_max_epochs: Math.max(100, Math.floor(Number(adminSettings.default_user_max_epochs ?? 500000))),
              max_running_per_user: Math.max(0, Math.min(64, Math.floor(Number(adminSettings.max_running_per_user ?? 1)))),
              apply_defaults_to_temp_users: !!syncExisting,
            }),
          });
          _applyAdminSettingsPayload(sRes);
          await Promise.all([loadAdminData(), refreshMeQuietly()]);
          if (syncExisting) {
            notify(t('msg.defQuotasSyncedToast', { count: Number(sRes.updated_temp_users || 0) }), 'success');
          } else {
            notify(t('msg.settingsSavedToast'), 'success');
          }
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          adminSettings.savingQuotas = false;
        }
      }

      async function saveAdminStorageSettings() {
        if (adminSettings.savingStorage) return;
        adminSettings.savingStorage = true;
        try {
          const sRes = await apiFetch('/api/v1/admin/settings', {
            method: 'PUT',
            body: JSON.stringify({
              data_cache_root: String(adminSettings.data_cache_root || '').trim(),
              migrate_existing: !!adminSettings.migrate_existing,
            }),
          });
          _applyAdminSettingsPayload(sRes);
          adminSettings.migrate_existing = false;
          const migCnt = Number(sRes.migrated_items || 0);
          if (migCnt > 0) {
            notify(t('msg.storagePathSavedMigratedToast', { count: migCnt }), 'success');
          } else {
            notify(t('msg.storagePathSavedToast'), 'success');
          }
          await Promise.all([fetchAssets(), fetchCheckpoints(), fetchQueueAndTasks(), loadAdminData()]);
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          adminSettings.savingStorage = false;
        }
      }

      async function resetAdminStorageDefault() {
        if (!confirm(t('msg.confirmResetStorageDefault'))) return;
        adminSettings.data_cache_root = '';
        await saveAdminStorageSettings();
      }

      async function exportAnalyticsDb(fmt = 'csv', allUsers = false) {
        try {
          const headers = {};
          if (currentLang.value) headers['X-WildCat-Lang'] = currentLang.value;
          if (clientFingerprint) headers['X-Client-FP'] = clientFingerprint;
          const safeRoam = (roamCodeLocal.value || '').replace(/[^\x20-\x7E]/g, '').trim();
          if (safeRoam) headers['X-Roam-Code'] = safeRoam;
          if (adminToken.value && (allUsers || topNav.value === 'admin')) {
            headers['X-Admin-Token'] = adminToken.value;
          }
          const q = `fmt=${encodeURIComponent(fmt || 'csv')}${allUsers ? '&all_users=true' : ''}`;
          const res = await fetch(`/api/v1/experiments/export-db?${q}`, { credentials: 'same-origin', headers });
          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || t('msg.reqFailedStatus', { status: res.status }));
          }
          const disp = res.headers.get('Content-Disposition') || '';
          const m = disp.match(/filename="?([^";]+)"?/i);
          const ext = String(fmt || 'csv').toLowerCase() === 'json' ? 'json' : 'csv';
          const fname = (m && m[1]) || `wildcat_analytics_${Date.now()}.${ext}`;
          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          triggerBrowserDownload(url, fname);
          setTimeout(() => URL.revokeObjectURL(url), 5000);
          notify(t('msg.analyticsExportedToast', { filename: fname }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminCreateUser() {
        try {
          const payload = {
            username: (newUser.username || '').trim(),
            note: (newUser.note || '').trim(),
            perm_lab: !!newUser.perm_lab,
            perm_api: !!newUser.perm_api,
          };
          const res = await apiFetch('/api/v1/admin/users', {
            method: 'POST',
            body: JSON.stringify(payload),
          });
          const createdName = (res.user && res.user.username) || newUser.username;
          notify(t('msg.userCreatedToast', { user: createdName }), 'success');
          newUser.username = '';
          newUser.note = '';
          newUser.perm_lab = false;
          newUser.perm_api = false;
          await loadAdminData();
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function impersonateUserFromAdmin(u) {
        if (!u || !u.id) return;
        try {
          const d = await apiFetch(`/api/v1/admin/impersonate/${u.id}`, { method: 'POST' });
          if (d && d.user) {
            _applyUserSession(d.user);
          }
          selectedDatasetId.value = null;
          selectedTaskId.value = null;
          tasks.value = [];
          topNav.value = 'studio';
          if (typeof window !== 'undefined' && window.history && window.location.pathname === '/admin') {
            window.history.pushState({}, '', '/');
          }
          await initWorkspace();
          notify(t('msg.switchedMachine', { machine: u.username }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminSaveUserQuota(u) {
        if (!u || !u.id) return;
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, {
            method: 'PUT',
            body: JSON.stringify({
              max_tasks: Math.max(0, Math.floor(Number(u.edit_max_tasks ?? u.max_tasks ?? 2))),
              max_epochs: Math.max(100, Math.floor(Number(u.edit_max_epochs ?? u.raw_max_epochs ?? 500000))),
            }),
          });
          await Promise.all([loadAdminData(), refreshMeQuietly()]);
          notify(t('msg.userUpdatedToast', { user: u.username }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminResetUserUsedTasks(u) {
        if (!u || !u.id) return;
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, {
            method: 'PUT',
            body: JSON.stringify({ reset_used_tasks: true }),
          });
          await Promise.all([loadAdminData(), refreshMeQuietly()]);
          notify(t('msg.userUpdatedToast', { user: u.username }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminToggleUserPerm(u, permField, rawCurrent) {
        if (!u || !u.id || u.role === 'admin') return;
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, {
            method: 'PUT',
            body: JSON.stringify({ [permField]: !rawCurrent }),
          });
          await Promise.all([loadAdminData(), refreshMeQuietly()]);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminCopyUserRoam(u) {
        if (!u || !u.recovery_code) return;
        await copyTextValue(u.recovery_code, 'msg.roamCodeCopiedToast');
      }

      async function adminRegenUserRoam(u) {
        if (!u || !u.id) return;
        if (!confirm(t('msg.confirmRegenRoam', { user: u.username }))) return;
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, {
            method: 'PUT',
            body: JSON.stringify({ regenerate_recovery_code: true }),
          });
          await Promise.all([loadAdminData(), refreshMeQuietly()]);
          notify(t('msg.userUpdatedToast', { user: u.username }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminPromptEditNote(u) {
        const n = prompt(t('msg.promptMachineNote', { machine: u.username }), u.note || '');
        if (n === null) return;
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, {
            method: 'PUT',
            body: JSON.stringify({ note: n }),
          });
          await loadAdminData();
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminToggleUserActive(u) {
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, {
            method: 'PUT',
            body: JSON.stringify({ is_active: !u.is_active }),
          });
          await loadAdminData();
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function adminDeleteUser(u) {
        if (!confirm(t('msg.confirmDeleteMachine', { machine: u.username }))) return;
        try {
          await apiFetch(`/api/v1/admin/users/${u.id}`, { method: 'DELETE' });
          notify(t('msg.machineDeletedToast', { machine: u.username }), 'warning');
          await loadAdminData();
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      // ───────── Formatting Helpers ─────────
      function statusBadgeText(taskObj) {
        if (!taskObj) return t('msg.badgeIdle');
        if (taskObj.status === 'running') return t('msg.badgeRunning', { gpu: taskObj.assigned_gpu ?? 0 });
        if (taskObj.status === 'queued') return t('msg.badgeQueued', { pos: taskObj.queue_position || 1 });
        if (taskObj.status === 'completed') return t('msg.badgeCompleted');
        if (taskObj.status === 'stopped') return t('msg.badgeStopped');
        if (taskObj.status === 'cancelled') return t('msg.badgeCancelled');
        if (taskObj.status === 'failed') return t('msg.badgeFailed');
        return t('msg.badgeIdle');
      }

      function statusLabel(taskObj) {
        if (!taskObj) return t('msg.badgeIdle');
        if (taskObj.status === 'running') return t('msg.statusRunning', { gpu: taskObj.assigned_gpu ?? 0 });
        if (taskObj.status === 'queued') {
          const ahead = taskAheadCount(taskObj);
          return t('msg.statusQueued', { pos: taskObj.queue_position || 1, ahead });
        }
        if (taskObj.status === 'completed') return t('msg.statusCompleted');
        if (taskObj.status === 'stopped') return t('msg.statusStopped', { step: taskObj.last_step || 0 });
        if (taskObj.status === 'cancelled') return t('msg.statusCancelled');
        if (taskObj.status === 'failed') return t('msg.statusFailed');
        return t('msg.badgeIdle');
      }

      function progressPct(taskObj) {
        if (!taskObj || !taskObj.total_steps) return '0.0';
        return Math.min(100, ((taskObj.last_step || 0) / taskObj.total_steps) * 100).toFixed(1);
      }

      // ───────── Experiment Lab (Batch Staging + Multi-Run Comparison + Sequence Flicker) ─────────
      const LAB_COLORS = [
        '#818cf8',
        '#34d399',
        '#fbbf24',
        '#f472b6',
        '#22d3ee',
        '#c084fc',
        '#fb923c',
        '#a3e635',
        '#38bdf8',
        '#f87171',
      ];

      const UNGROUPED_LAB_KEY = '__ungrouped__';

      function createEmptyStagedGroup(idx = 1, customName = '') {
        return {
          gid: `sgrp_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          name: customName || '',
          collapsed: false,
          targetBatchId: '',
          items: [],
        };
      }

      function loadInitialLabStagedGroups() {
        try {
          if (typeof localStorage !== 'undefined') {
            const rawV2 = localStorage.getItem('wc_lab_staged_groups_v2');
            if (rawV2) {
              const parsedV2 = JSON.parse(rawV2);
              if (Array.isArray(parsedV2) && parsedV2.length > 0) {
                return parsedV2.map((g, i) => ({
                  gid: String(g.gid || `sgrp_init_${i}`),
                  name: String(g.name || ''),
                  collapsed: !!g.collapsed,
                  targetBatchId: String(g.targetBatchId || ''),
                  items: Array.isArray(g.items) ? g.items : [],
                }));
              }
            }
            const rawV1 = localStorage.getItem('wc_lab_staged_v1');
            if (rawV1) {
              const parsedV1 = JSON.parse(rawV1);
              if (Array.isArray(parsedV1) && parsedV1.length > 0) {
                const grp = createEmptyStagedGroup(1);
                grp.items = parsedV1;
                return [grp];
              }
            }
          }
        } catch (_) {}
        return [createEmptyStagedGroup(1)];
      }

      function loadInitialCollapsedLabGroups() {
        try {
          if (typeof localStorage !== 'undefined') {
            const raw = localStorage.getItem('wc_lab_collapsed_groups');
            if (raw) {
              const p = JSON.parse(raw);
              if (p && typeof p === 'object') return p;
            }
          }
        } catch (_) {}
        return {};
      }

      const labStagedGroups = ref(loadInitialLabStagedGroups());
      const labActiveStagedGroupId = ref(labStagedGroups.value[0].gid);
      const labNote = ref('');
      const labBatchName = ref('');
      const labSubmitting = ref(false);
      const labActiveBatchId = ref(
        (typeof localStorage !== 'undefined' ? localStorage.getItem('wc_lab_active_batch') : '') || ''
      );

      const labSelectedTaskIds = ref([]);
      const labBaselineTaskId = ref(null);
      const labTaskFilter = ref('');
      const labTaskSortMode = ref('manual'); // 'manual' | 'psnr_desc' | 'hfi_asc' | 'val_asc'
      const labArchiving = ref(false);
      const labCollapsedGroups = reactive(loadInitialCollapsedLabGroups());
      const labGroupCreateOpen = ref(false);
      const labNewGroupName = ref('');
      const labCreateMoveSelected = ref(true);
      const labRenamingGroupId = ref('');
      const labRenamingGroupName = ref('');

      const labEvalAssetId = ref(null);
      const labCkptMode = ref('best'); // 'best' | 'latest'
      const labCompare = reactive({
        loading: false,
        runs: [],
        eval_asset: null,
        eval_runner: {
          busy: false,
          current_task_id: null,
          current_node: '',
          phase: '',
          queued_tasks: 0,
          task_done: 0,
          task_total: 0,
          frame_done: 0,
          frame_total: 0,
          last_error: '',
        },
      });

      const labChartTab = ref('train_loss'); // 'train_loss' | 'val_mse' | 'seq_hfi' | 'seq_delta' | 'seq_psnr'
      const labChartCfg = reactive({ logScale: true, smooth: 0.55 });
      const labCompareCanvas = ref(null);
      const labChartHover = ref(null);
      let labChartPlotMeta = null;

      const labViewer = reactive({
        mode: 'out', // 'out' | 'flicker' | 'err'
        showSource: true,
        index: 0,
        playing: false,
        fps: 12,
        zoom: 1,
        panX: 0,
        panY: 0,
        dragging: false,
        dragStartX: 0,
        dragStartY: 0,
      });

      const activeStagedGroup = computed(() => {
        if (!labStagedGroups.value.length) {
          labStagedGroups.value.push(createEmptyStagedGroup(1));
        }
        return (
          labStagedGroups.value.find((g) => g.gid === labActiveStagedGroupId.value) ||
          labStagedGroups.value[0]
        );
      });

      // Backward-compatible computed flattened staged list
      const labStaged = computed(() => {
        const out = [];
        for (const g of labStagedGroups.value) {
          if (Array.isArray(g.items)) out.push(...g.items);
        }
        return out;
      });

      const labTotalStagedCount = computed(() => labStaged.value.length);

      const labNonEmptyStagedGroupsCount = computed(() => {
        return labStagedGroups.value.filter((g) => Array.isArray(g.items) && g.items.length > 0).length;
      });

      watch(
        labStagedGroups,
        (val) => {
          try {
            if (typeof localStorage !== 'undefined') {
              localStorage.setItem('wc_lab_staged_groups_v2', JSON.stringify(val || []));
              localStorage.removeItem('wc_lab_staged_v1');
            }
          } catch (_) {}
        },
        { deep: true }
      );

      watch(
        labCollapsedGroups,
        (val) => {
          try {
            if (typeof localStorage !== 'undefined') {
              localStorage.setItem('wc_lab_collapsed_groups', JSON.stringify(val || {}));
            }
          } catch (_) {}
        },
        { deep: true }
      );

      watch(labActiveBatchId, (val) => {
        try {
          if (typeof localStorage !== 'undefined' && val) {
            localStorage.setItem('wc_lab_active_batch', val);
          }
        } catch (_) {}
      });

      function summarizeClientConfig(cfg) {
        const c = cfg || {};
        const mo = c.model || {};
        const tr = c.train || {};
        const av = c.advanced || {};
        const tags = [];
        let nonDefault = false;

        const msize = mo.model_size || 'Medium';
        if (msize !== 'Medium') {
          nonDefault = true;
          tags.push(msize);
        }
        if (tr.multi_resolution === false) {
          nonDefault = true;
          tags.push('MultiRes=Off');
        }
        const prec = String(tr.precision || 'adaptive').toLowerCase();
        if (prec !== 'adaptive') {
          nonDefault = true;
          tags.push(prec.toUpperCase());
        }
        if (tr.batch_size_auto === false) {
          nonDefault = true;
          tags.push(`Batch=${tr.batch_size || 8}`);
        }
        const crop = Number(tr.crop_size || 256);
        if (crop !== 256) {
          nonDefault = true;
          tags.push(`Crop=${crop}`);
        }
        const ep = Number(tr.epochs || 0);
        if (ep > 0 && ep !== 10000) {
          tags.push(`${ep}ep`);
        }
        const iw = String(tr.initial_weights || 'None').trim();
        if (iw && iw.toLowerCase() !== 'none') {
          nonDefault = true;
          const shortIw = iw.split('/').pop().replace(/\.(pt|cat)$/i, '');
          tags.push(`Init=${shortIw}`);
        }
        const multiRes = tr.multi_resolution !== false;
        const cIn = Number(mo.in_channels || 3);
        const cOut = Number(mo.out_channels || 3);
        if (av.crop_grid && av.crop_grid.enabled && (Number(av.crop_grid.full_res_grid || 4) !== 4 || av.crop_grid.anchor === 'top')) {
          nonDefault = true;
          const gTop = Number(av.crop_grid.full_res_grid || 4) !== 1 && av.crop_grid.anchor === 'top' ? '·Top' : '';
          tags.push(`Grid=${av.crop_grid.full_res_grid}px${gTop}`);
        }
        if (av.loss_family && av.loss_family.enabled) {
          const lp = [];
          if (Number(av.loss_family.charbonnier_ratio) > 0) lp.push(`Charb(${av.loss_family.charbonnier_ratio})`);
          if (Number(av.loss_family.grad_weight) > 0) lp.push(`${av.loss_family.grad_weight}∇`);
          if (cOut !== 1 && Number(av.loss_family.hdr_alpha) > 0) lp.push(`HDR(${av.loss_family.hdr_alpha})`);
          if (lp.length) {
            nonDefault = true;
            tags.push(lp.join('+'));
          }
        }
        if (multiRes && av.stage_schedule && av.stage_schedule.enabled) {
          nonDefault = true;
          tags.push(
            `Stage(${Math.round(av.stage_schedule.r_quarter || 15)}/${Math.round(av.stage_schedule.r_half || 25)}/${Math.round(av.stage_schedule.r_full || 60)})`
          );
        }
        if (av.ema_lr && av.ema_lr.enabled) {
          nonDefault = true;
          tags.push(`EMA(${av.ema_lr.ema_decay})`);
        }
        if (av.augment && av.augment.enabled) {
          const ap = [];
          if (Number(av.augment.hflip_prob) > 0) ap.push(`Flip(${av.augment.hflip_prob})`);
          if (!(cIn < 3 && cOut < 3) && Number(av.augment.exposure_ev) > 0) ap.push(`EV±${av.augment.exposure_ev}`);
          if (ap.length) {
            nonDefault = true;
            tags.push(`Aug(${ap.join('/')})`);
          }
        }
        if (av.kf_common && av.kf_common.enabled && cIn === 3 && cOut === 3) {
          // same tag as backend/experiments.py summarize_config_diff
          const kc = clampKfCommon(av.kf_common);
          nonDefault = true;
          tags.push(`KfCommon(thr=${kc.threshold},span=${kc.span}${kc.flow === 'raft' ? ',raft' : ''}${kc.overall ? '' : ',no-overall'})`);
        }
        return { is_nuke_native: !nonDefault, model_size: msize, tags };
      }

      function nextUniqueExpNodeName(hint = '') {
        const taken = new Set([
          ...tasks.value.map((x) => x.node_name),
          ...labStaged.value.map((x) => x.node_name),
        ]);
        const cleanHint = String(hint || '')
          .replace(/[^A-Za-z0-9_-]/g, '')
          .slice(0, 14);
        let idx = labStaged.value.length + 1;
        while (true) {
          const cand = cleanHint ? `Exp${idx}_${cleanHint}` : `Exp_${idx}`;
          if (!taken.has(cand)) return cand;
          idx++;
        }
      }

      function prepareFormForStaging() {
        if (selectedDataset.value) {
          form.source = selectedDataset.value.source_path || selectedDataset.value.rel_path || '';
          form.target = selectedDataset.value.target_path || selectedDataset.value.rel_path || '';
        }
        if (!isChannelsLocked.value) {
          onChannelPresetChange();
        }
        if (!selectedDataset.value || !form.source || !form.target) {
          notify(t('msg.errSelectDatasetFirst'), 'error');
          return false;
        }
        if (initCategory.value === 'checkpoint' && !form.checkpoint_path) {
          notify(t('msg.errSelectCheckpointFirst'), 'error');
          return false;
        }
        const dsIn = Number(selectedDataset.value.in_channels || 0);
        const dsOut = Number(selectedDataset.value.out_channels || 0);
        const reqIn = Number(form.in_channels || 3);
        const reqOut = Number(form.out_channels || 3);
        if (dsIn > 0 && dsOut > 0 && (dsIn < reqIn || (reqOut > 1 && dsOut < reqOut))) {
          notify(
            t('msg.errDatasetChannelMismatch', {
              dsIn,
              dsOut,
              reqIn,
              reqOut,
            }),
            'error'
          );
          return false;
        }
        const epVal = Math.floor(Number(form.epochs || 0));
        if (!Number.isFinite(epVal) || epVal < 1) {
          notify(t('msg.errInvalidEpochs'), 'error');
          return false;
        }
        if (userMaxEpochs.value > 0 && epVal > userMaxEpochs.value) {
          notify(t('msg.errTempMaxEpochs', { max: userMaxEpochs.value, req: epVal }), 'error');
          return false;
        }
        form.epochs = epVal;
        if (form.batch_size_mode === 'Manual') {
          const bsVal = Math.floor(Number(form.batch_size || 0));
          if (!Number.isFinite(bsVal) || bsVal < 1) {
            notify(t('msg.errInvalidBatch'), 'error');
            return false;
          }
          form.batch_size = bsVal;
        } else {
          form.batch_size = 8;
        }
        return true;
      }

      function addNewLabStagedGroup(customName = '') {
        const grp = createEmptyStagedGroup(labStagedGroups.value.length + 1, customName);
        labStagedGroups.value.push(grp);
        labActiveStagedGroupId.value = grp.gid;
        return grp;
      }

      function removeLabStagedGroup(gid) {
        if (labStagedGroups.value.length <= 1) {
          labStagedGroups.value[0].items = [];
          labStagedGroups.value[0].name = '';
          labStagedGroups.value[0].targetBatchId = '';
          return;
        }
        const idx = labStagedGroups.value.findIndex((g) => g.gid === gid);
        if (idx >= 0) {
          labStagedGroups.value.splice(idx, 1);
          if (labActiveStagedGroupId.value === gid) {
            labActiveStagedGroupId.value = labStagedGroups.value[Math.max(0, idx - 1)].gid;
          }
        }
      }

      function moveLabStagedGroupOrder(gIdx, dir) {
        const target = gIdx + dir;
        if (target < 0 || target >= labStagedGroups.value.length) return;
        const arr = labStagedGroups.value;
        const tmp = arr[gIdx];
        arr[gIdx] = arr[target];
        arr[target] = tmp;
      }

      function moveLabStagedItemOrder(gid, itemIdx, dir) {
        const grp = labStagedGroups.value.find((g) => g.gid === gid);
        if (!grp || !Array.isArray(grp.items)) return;
        const target = itemIdx + dir;
        if (target < 0 || target >= grp.items.length) return;
        const tmp = grp.items[itemIdx];
        grp.items[itemIdx] = grp.items[target];
        grp.items[target] = tmp;
      }

      function moveLabStagedItemToGroup(fromGid, uid, rawToGid, selectEl = null) {
        if (selectEl) selectEl.value = '';
        if (!rawToGid || rawToGid === fromGid) return;
        const srcGrp = labStagedGroups.value.find((g) => g.gid === fromGid);
        if (!srcGrp || !Array.isArray(srcGrp.items)) return;
        const idx = srcGrp.items.findIndex((x) => x.uid === uid);
        if (idx < 0) return;
        const [moved] = srcGrp.items.splice(idx, 1);
        let dstGrp = null;
        if (rawToGid === '__new__') {
          dstGrp = addNewLabStagedGroup('');
        } else {
          dstGrp = labStagedGroups.value.find((g) => g.gid === rawToGid);
        }
        if (dstGrp) {
          dstGrp.collapsed = false;
          dstGrp.items.push(moved);
          labActiveStagedGroupId.value = dstGrp.gid;
        }
      }

      function addCurrentToLabList() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        if (!prepareFormForStaging()) return;
        const cfg = buildConfigPayload(false);
        const diff = summarizeClientConfig(cfg);
        const taken = new Set([
          ...tasks.value.map((x) => x.node_name),
          ...labStaged.value.map((x) => x.node_name),
        ]);
        let nodeName = (form.run_name || '').trim();
        if (!nodeName || taken.has(nodeName)) {
          nodeName = nextUniqueExpNodeName(diff.is_nuke_native ? 'NukeBase' : diff.model_size);
        }
        cfg.run_name = nodeName;
        const item = {
          uid: `st_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          node_name: nodeName,
          preferred_gpu: form.preferred_gpu || 'auto',
          dataset_name: selectedDataset.value ? selectedDataset.value.name : '',
          dataset_id: selectedDataset.value ? selectedDataset.value.id : null,
          note: (labNote.value || '').trim(),
          diff,
          config: cfg,
          formSnap: JSON.parse(JSON.stringify(form)),
          advSnap: JSON.parse(JSON.stringify(adv)),
          initCategorySnap: initCategory.value,
        };
        const grp = activeStagedGroup.value;
        grp.collapsed = false;
        grp.items.push(item);
        labNote.value = '';
        form.run_name = nextUniqueExpNodeName();
        notify(t('msg.labAddedToast', { name: nodeName }), 'success');
      }

      function addNukeBaselineToLabList() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        if (!prepareFormForStaging()) return;
        const baseCfg = buildConfigPayload(false);
        delete baseCfg.advanced;
        baseCfg.model.model_size = 'Medium';
        baseCfg.train.multi_resolution = true;
        baseCfg.train.precision = 'adaptive';
        baseCfg.train.batch_size_auto = true;
        baseCfg.train.batch_size = 8;
        baseCfg.train.crop_size = 256;
        baseCfg.train.initial_weights = 'None';
        const nFrames = Math.max(1, Number((selectedDataset.value && selectedDataset.value.frame_count) || 1));
        const bs = Math.min(nFrames, 8);
        baseCfg.train.total_steps = Math.max(1, Math.ceil((baseCfg.train.epochs * nFrames) / bs));
        const nodeName = nextUniqueExpNodeName('NukeBase');
        baseCfg.run_name = nodeName;
        const diff = summarizeClientConfig(baseCfg);
        const formSnap = JSON.parse(JSON.stringify(form));
        formSnap.model_size = 'Medium';
        formSnap.multi_resolution = true;
        formSnap.precision = 'adaptive';
        formSnap.batch_size_mode = 'Auto';
        formSnap.batch_size = 8;
        formSnap.crop_size = 256;
        formSnap.initial_weights_mode = 'None';
        formSnap.checkpoint_path = '';
        const grp = activeStagedGroup.value;
        grp.collapsed = false;
        grp.items.unshift({
          uid: `st_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          node_name: nodeName,
          preferred_gpu: form.preferred_gpu || 'auto',
          dataset_name: selectedDataset.value ? selectedDataset.value.name : '',
          dataset_id: selectedDataset.value ? selectedDataset.value.id : null,
          note: t('lab.nukeNativeTag'),
          diff,
          config: baseCfg,
          formSnap,
          advSnap: createDefaultAdvConfig(),
          initCategorySnap: 'scratch',
        });
        notify(t('msg.labAddedToast', { name: nodeName }), 'success');
      }

      // ───────── Recommended Best-Combo Exploration Generator (Parameter Relaxation Suite) ─────────
      const VALID_COMBO_TASK_TYPES = [
        'cleanup_grain',
        'keying_despill',
        'bokeh_focus',
        'degrain_clean',
        'roto_matte',
        'face_edit',
      ];

      const labComboCfg = reactive({
        panelOpen: false,
        previewOpen: true,
        taskType: 'auto', // 'auto' | 'cleanup_grain' | 'keying_despill' | 'bokeh_focus' | 'degrain_clean' | 'roto_matte' | 'face_edit'
        scope: 'standard', // 'compact' (4) | 'standard' (6) | 'thorough' (8)
        groupName: '',
      });

      const hasExplicitDatasetTaskType = computed(() => {
        const ds = selectedDataset.value;
        return !!(ds && ds.task_type && VALID_COMBO_TASK_TYPES.includes(ds.task_type));
      });

      const detectedLabComboTaskType = computed(() => {
        const ds = selectedDataset.value;
        if (!ds) return 'cleanup_grain';
        if (ds.task_type && VALID_COMBO_TASK_TYPES.includes(ds.task_type)) {
          return ds.task_type;
        }
        const ic = Number(ds.in_channels || 3);
        const oc = Number(ds.out_channels || 3);
        const hint = `${ds.name || ''} ${ds.demo_key || ''}`.toLowerCase();
        if (oc === 1) return 'roto_matte';
        if (ic === 3 && oc === 4) return 'keying_despill';
        if (/bokeh|defocus|blur|focus|dof|motionblur/.test(hint)) return 'bokeh_focus';
        if (/key|green|blue|spill|despill/.test(hint) && oc >= 4) return 'keying_despill';
        if (/degrain|denoise|noise|cleanplate|smartvector|marker|dark/.test(hint)) return 'degrain_clean';
        if (/roto|mask|matte|alpha/.test(hint) && oc === 1) return 'roto_matte';
        // face edits: RGB -> RGB and a name that says so (checked after the more specific names above, before the fallbacks)
        if (ic === 3 && oc === 3 && /face|wound|scar|beauty|aging|deag|swap|makeup/.test(hint)) return 'face_edit';
        if (Number(ds.avg_mae || 0) > 0 && Number(ds.avg_mae) < 0.012) return 'degrain_clean';
        return 'cleanup_grain';
      });

      const effectiveLabComboTaskType = computed(() => {
        return labComboCfg.taskType === 'auto' ? detectedLabComboTaskType.value : labComboCfg.taskType;
      });

      function _buildComboCandidateRecipes(ds, effType, scope) {
        const dsIn = Number((ds && ds.in_channels) || form.in_channels || 3);
        const dsOut = Number((ds && ds.out_channels) || form.out_channels || 3);
        let cIn = 3;
        let cOut = 3;
        if (isChannelsLocked.value) {
          cIn = Number(form.in_channels || 3);
          cOut = Number(form.out_channels || 3);
        } else if (dsOut === 1) {
          cIn = dsIn >= 4 ? 4 : 3;
          cOut = 1;
        } else if (effType === 'roto_matte' && dsIn >= 3) {
          cIn = 3;
          cOut = 1;
        } else if (effType === 'keying_despill' && dsOut >= 4) {
          cIn = dsIn >= 4 ? 4 : 3;
          cOut = 4;
        } else if (effType === 'face_edit' && dsIn >= 3 && dsOut >= 3) {
          // the face-edit preset is RGB -> RGB (keyframe commonality needs it), also on a dataset that carries an alpha
          [cIn, cOut] = FACE_EDIT_PRESET.channels.split('->').map(Number);
        } else if (dsIn >= 4 && dsOut >= 4) {
          cIn = 4;
          cOut = 4;
        }

        const minSide =
          ds && Number(ds.width) >= 32 && Number(ds.height) >= 32
            ? Math.min(Number(ds.width), Number(ds.height))
            : 1080;
        const maxLegalCrop = Math.max(32, Math.min(512, Math.floor(minSide / 16) * 16));
        const clampCrop = (c) => Math.max(32, Math.min(maxLegalCrop, Math.floor(c / 16) * 16));
        const cropTight = clampCrop(minSide < 256 ? 192 : 256);
        const cropWide = clampCrop(minSide < 720 ? 256 : 384);
        const cropMax = clampCrop(minSide >= 1024 ? 512 : cropWide);

        const pickModel = (preferred) => (isModelSizeLocked.value ? form.model_size || 'Medium' : preferred);
        const clampEp = (ep) => {
          const clean = Math.max(100, Math.round(Number(ep) || 150000));
          return userMaxEpochs.value > 0 ? Math.min(clean, userMaxEpochs.value) : clean;
        };

        // Recipes follow the 2026-11 re-validation (help center, strategy evidence): the anchor of each task type is its preset,
        // every other recipe changes ONE thing, and the last one is the CopyCat default as control (same Model Size as the
        // anchor). Charbonnier appears where its tests hold (face-edit preset, a fidelity recipe for clean-up); HDR damping and
        // horizontal flip only in the "previous preset" recipe kept for comparison.
        // All recipes of a test use the user's current Epochs (equal lengths compare fairly).
        const comboEp = clampEp(Number(form.epochs) || 100000);
        const G05 = { cr: 0, gw: 0.05, ha: 0 };
        const STG = [15, 25, 60];
        const mk = (code, role, reasonKey, o = {}) => ({
          code,
          role,
          reasonKey,
          nukeNative: !!o.native,
          model_size: pickModel(o.m || 'Small'),
          crop_size: o.native ? clampCrop(256) : o.crop || cropTight,
          grid: o.grid || 1,
          loss: o.loss || null,
          stage: o.stage || null,
          aug: o.aug || null,
          kf: !!o.kf,
          kfOverall: o.kfOverall !== false,
          kfFlow: o.kfFlow === 'raft' ? 'raft' : 'dis',
          ema: !!o.ema,
          epochs: o.epochs ? clampEp(o.epochs) : comboEp,
        });
        const control = (m) => mk('NukeBase', 'control', 'lab.comboReason_nuke', { native: true, m });

        // keyframe commonality can only take effect on RGB -> RGB with the plate attached and at least 3 keyframes; RAFT
        // is offered only where the server can run it right now
        const kfOk = !kfCommonBlocker(cIn, cOut, ds);
        const raftOk = !!kfCommonRaftOffered.value;
        let specs = [];
        if (effType === 'face_edit') {
          // face edits: anchor = the preset (1px + Charbonnier 0.25 + keyframe commonality + averaged weights); every other
          // recipe changes one thing of it; the CopyCat default closes the list. Where keyframe commonality cannot take
          // effect, the recipes with it would repeat the ones without it: they are left out and the panel says why.
          const m = FACE_EDIT_PRESET.model_size;
          const FL = FACE_EDIT_PRESET.loss;
          const P = { m, crop: clampCrop(FACE_EDIT_PRESET.crop), ema: true, loss: { cr: FL.charbonnier_ratio, gw: FL.grad_weight, ha: FL.hdr_alpha } };
          specs = kfOk
            ? [
                mk('Anchor_Preset', 'anchor', 'lab.comboReason_fe_1', Object.assign({}, P, { kf: true })),
                mk('No_KfCommon', 'relaxed', 'lab.comboReason_fe_2', P),
                mk('No_Charb', 'relaxed', 'lab.comboReason_fe_8', Object.assign({}, P, { kf: true, loss: null })),
                mk('Small_Preset', 'relaxed', 'lab.comboReason_fe_4', Object.assign({}, P, { m: 'Small', kf: true })),
                mk('Kf_NoOverall', 'relaxed', 'lab.comboReason_fe_5', Object.assign({}, P, { kf: true, kfOverall: false })),
              ]
                .concat(raftOk ? [mk('Kf_Raft', 'relaxed', 'lab.comboReason_fe_6', Object.assign({}, P, { kf: true, kfFlow: 'raft' }))] : [])
                .concat([mk('Kf_Grid4', 'relaxed', 'lab.comboReason_fe_3', Object.assign({}, P, { kf: true, grid: 4 })), control(m)])
            : [
                mk('Anchor_NoKf', 'anchor', 'lab.comboReason_fe_nokf', P),
                mk('Small_NoKf', 'relaxed', 'lab.comboReason_fe_7', Object.assign({}, P, { m: 'Small' })),
                mk('No_Charb', 'relaxed', 'lab.comboReason_fe_9', Object.assign({}, P, { loss: null })),
                control(m),
              ];
        } else if (effType === 'cleanup_grain') {
          // paint / erase shots: hand-painted keyframes can disagree like face edits do -> keyframe commonality is offered
          // as one recipe where it can take effect (its evidence comes from face shots)
          const A = { m: PRESET_SPECS.cleanup_grain.model_size, loss: G05, ema: true, stage: STG };
          specs = [mk('Anchor_Preset', 'anchor', 'lab.comboReason_cg_1', A)]
            .concat(kfOk ? [mk('Kf_Common', 'relaxed', 'lab.comboReason_cg_kf', Object.assign({}, A, { kf: true }))] : [])
            .concat([
              mk('Grid1_Only', 'relaxed', 'lab.comboReason_cg_2', { m: A.m, ema: true, stage: STG }),
              mk('Small_Preset', 'relaxed', 'lab.comboReason_cg_3', Object.assign({}, A, { m: 'Small' })),
              mk('Charb_Fidelity', 'relaxed', 'lab.comboReason_cg_8', { m: A.m, ema: true, stage: STG, loss: { cr: 0.25, gw: 0.05, ha: 0 } }),
              mk('Wide_Crop', 'relaxed', 'lab.comboReason_cg_6', Object.assign({}, A, { crop: cropWide })),
              // the previous preset (until Patch 58): equal stage thirds, everything else as recommended
              mk('Prev_Preset', 'relaxed', 'lab.comboReason_cg_7', { m: A.m, loss: G05, ema: true }),
              control(A.m),
            ]);
        } else if (effType === 'keying_despill') {
          specs = [
            mk('Anchor_Preset', 'anchor', 'lab.comboReason_kd_1', { loss: { cr: 0, gw: 0.15, ha: 0 }, ema: true }),
            mk('Grid1_Only', 'relaxed', 'lab.comboReason_kd_2', { ema: true }),
            mk('Grad_005', 'relaxed', 'lab.comboReason_kd_3', { loss: G05, ema: true }),
            mk('Med_Preset', 'relaxed', 'lab.comboReason_kd_4', { m: 'Medium', loss: { cr: 0, gw: 0.15, ha: 0 }, ema: true }),
            mk('Grad_030', 'relaxed', 'lab.comboReason_kd_5', { loss: { cr: 0, gw: 0.3, ha: 0 }, ema: true }),
            mk('Wide_Crop', 'relaxed', 'lab.comboReason_kd_6', { loss: { cr: 0, gw: 0.15, ha: 0 }, crop: cropWide, ema: true }),
            mk('Prev_Preset', 'relaxed', 'lab.comboReason_kd_7', { loss: { cr: 0.3, gw: 0.15, ha: cOut === 1 ? 0 : 0.35 } }),
            control('Small'),
          ];
        } else if (effType === 'bokeh_focus') {
          specs = [
            mk('Anchor_Preset', 'anchor', 'lab.comboReason_bf_1', {
              m: 'Large',
              loss: { cr: 0.25, gw: 0.1, ha: cOut === 1 ? 0 : 0.5 },
              stage: STG,
            }),
            mk('Stage_Only', 'relaxed', 'lab.comboReason_bf_2', { m: 'Large', stage: STG }),
            mk('Med_Stage', 'relaxed', 'lab.comboReason_bf_3', { m: 'Medium', stage: STG }),
            mk('Equal_Stages', 'relaxed', 'lab.comboReason_bf_4', { m: 'Large' }),
            mk('Small_Stage', 'relaxed', 'lab.comboReason_bf_5', { stage: STG }),
            mk('Wide_Crop', 'relaxed', 'lab.comboReason_bf_6', { m: 'Large', stage: STG, crop: cropWide }),
            mk('Small_Grid1', 'relaxed', 'lab.comboReason_bf_7'),
            control('Large'),
          ];
        } else if (effType === 'degrain_clean') {
          specs = [mk('Anchor_Preset', 'anchor', 'lab.comboReason_dg_1')]
            // degrain gets slightly worse with longer training: offer the 100k version when the user trains longer
            .concat(comboEp > 100000 ? [mk('Short_100k', 'relaxed', 'lab.comboReason_dg_7', { epochs: 100000 })] : [])
            .concat([
            mk('Grad_005', 'relaxed', 'lab.comboReason_dg_2', { loss: G05 }),
            mk('Med_Grid1', 'relaxed', 'lab.comboReason_dg_3', { m: 'Medium' }),
            mk('Prev_Preset', 'relaxed', 'lab.comboReason_dg_4', { loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 } }),
            mk('Wide_Crop', 'relaxed', 'lab.comboReason_dg_5', { crop: cropWide }),
            mk('Grad_015', 'relaxed', 'lab.comboReason_dg_6', { loss: { cr: 0, gw: 0.15, ha: 0 } }),
            control('Small'),
          ]);
        } else {
          // roto_matte: no knob had a measurable effect in the re-validation, so the list is short on purpose
          specs = [
            mk('Anchor_Preset', 'anchor', 'lab.comboReason_rm_1'),
            mk('Prev_Preset', 'relaxed', 'lab.comboReason_rm_2', { loss: { cr: 0.25, gw: 0.05, ha: 0 } }),
            mk('Wide_Crop', 'relaxed', 'lab.comboReason_rm_3', { crop: cropWide }),
            mk('Grad_005', 'relaxed', 'lab.comboReason_rm_4', { loss: G05 }),
            control('Small'),
          ];
        }

        // a variant that trains exactly what an earlier recipe trains (e.g. "wide crop" on a plate whose short side is below
        // 720 px, where the wide crop is the same 256 px as the anchor's) would only repeat a run: it is left out
        const sig = (r) => JSON.stringify([r.nukeNative, r.model_size, r.crop_size, r.grid, r.loss, r.stage, r.aug, r.kf, r.kf && r.kfOverall, r.kf && r.kfFlow, r.ema, r.epochs]);
        const seenSig = new Set();
        specs = specs.filter((r, i) => {
          if (i === specs.length - 1) return true; // the control always stays
          const k = sig(r);
          if (seenSig.has(k)) return false;
          seenSig.add(k);
          return true;
        });
        const count = scope === 'compact' ? 4 : scope === 'thorough' ? 8 : 6;
        const controlSpec = specs[specs.length - 1];
        const chosen = count >= specs.length ? specs : specs.slice(0, count - 1).concat([controlSpec]);

        const nFrames = Math.max(1, Number((ds && ds.frame_count) || 8));
        const reqBs = form.batch_size_mode === 'Auto' ? 8 : Math.max(1, Math.floor(Number(form.batch_size) || 8));
        const effBs = Math.min(nFrames, reqBs);

        let iw = 'None';
        if (initCategory.value === 'checkpoint') {
          iw = form.checkpoint_path || 'None';
        }

        const srcPath = ds ? ds.source_path || ds.rel_path || '' : (form.source || '').trim();
        const tgtPath = ds ? ds.target_path || ds.rel_path || '' : (form.target || '').trim();

        return chosen.map((sp, idx) => {
          const ep = sp.epochs;
          const totSteps = Math.max(1, Math.ceil((ep * nFrames) / effBs));
          const advSnap = createDefaultAdvConfig();
          if (!sp.nukeNative) {
            if (sp.grid && sp.grid !== 4) {
              advSnap.crop_grid.enabled = true;
              advSnap.crop_grid.full_res_grid = sp.grid;
            }
            if (sp.loss && (sp.loss.cr > 0 || sp.loss.gw > 0 || sp.loss.ha > 0)) {
              advSnap.loss_family.enabled = true;
              advSnap.loss_family.charbonnier_ratio = sp.loss.cr;
              advSnap.loss_family.grad_weight = sp.loss.gw;
              advSnap.loss_family.hdr_alpha = cOut === 1 ? 0 : sp.loss.ha;
            }
            if (sp.stage && sp.stage.length === 3) {
              advSnap.stage_schedule.enabled = true;
              advSnap.stage_schedule.r_quarter = sp.stage[0];
              advSnap.stage_schedule.r_half = sp.stage[1];
              advSnap.stage_schedule.r_full = sp.stage[2];
              advSnap.stage_schedule.warmup_steps = 200;
            }
            if (sp.aug && (sp.aug.flip > 0 || sp.aug.ev > 0)) {
              advSnap.augment.enabled = true;
              advSnap.augment.hflip_prob = sp.aug.flip;
              advSnap.augment.exposure_ev = cIn < 3 && cOut < 3 ? 0 : sp.aug.ev;
            }
            if (sp.kf) {
              Object.assign(advSnap.kf_common, FACE_EDIT_PRESET.kf_common, { enabled: true, overall: sp.kfOverall, flow: sp.kfFlow });
            }
            if (sp.ema) {
              advSnap.ema_lr.enabled = true;
              advSnap.ema_lr.ema_decay = 0.9995;
            }
          }

          const cfg = {
            run_name: `Combo${idx + 1}_${sp.code}`,
            paths: {
              source: srcPath,
              target: tgtPath,
            },
            color: {
              working_colorspace: form.working_colorspace || 'ACEScg',
              source_colorspace: 'auto',
              target_colorspace: 'auto',
              output_colorspace: form.working_colorspace || 'ACEScg',
              output_format: form.output_format || 'exr_half',
              view_transform: 'ACES 1.0 - SDR Video',
            },
            model: {
              model_size: sp.model_size,
              in_channels: cIn,
              out_channels: cOut,
              n_levels: 3,
              scale: 0.5,
            },
            train: {
              epochs: ep,
              total_steps: totSteps,
              batch_size_auto: form.batch_size_mode === 'Auto',
              batch_size: effBs,
              crop_size: sp.crop_size,
              learning_rate: Number(form.learning_rate || 0.0001),
              lr_power: 0.3,
              multi_resolution: true,
              precision: form.precision || 'adaptive',
              seed: Number(form.seed || 0),
              initial_weights: sp.nukeNative ? 'None' : iw,
              resume: false,
            },
            optimizations: {
              engine: accelAny.value ? 'fast' : 'reference',
              lossless: {
                gpu_resident: form.opt_gpu_resident ? 'auto' : false,
                cuda_graph: !!(form.opt_cuda_graph && form.opt_fused_adam),
                fused_adam: !!form.opt_fused_adam,
                channels_last: !!form.opt_channels_last,
                cudnn_benchmark: !!form.opt_cudnn_benchmark,
                async_io: !!form.opt_async_io,
                preview_gpu: form.preview_gpu || 'auto',
              },
            },
            loss: { l1_weight: 0.0, l2_weight: 1.0 },
            io: {
              checkpoint_interval: Math.max(10, Number(form.checkpoint_interval || 1000)),
              preview_enabled: form.preview_gpu !== 'off',
              preview_interval: Math.max(10, Number(form.preview_interval || 100)),
              preview_size: 0,
              log_interval: Math.max(1, Number(form.log_interval || 10)),
            },
          };
          if (!sp.nukeNative) {
            cfg.advanced = JSON.parse(JSON.stringify(advSnap));
          }
          if (ds && ds.id) {
            cfg.experiment = { dataset_id: ds.id };
            if (ds.has_eval_sequence && ds.seq_path) {
              cfg.paths.eval_sequence = ds.seq_path;
            }
          }

          const formSnap = JSON.parse(JSON.stringify(form));
          formSnap.source = srcPath;
          formSnap.target = tgtPath;
          formSnap.model_size = sp.model_size;
          formSnap.in_channels = cIn;
          formSnap.out_channels = cOut;
          formSnap.epochs = ep;
          formSnap.crop_size = sp.crop_size;
          formSnap.multi_resolution = true;
          if (sp.nukeNative) {
            formSnap.initial_weights_mode = 'None';
            formSnap.checkpoint_path = '';
          }

          const diff = summarizeClientConfig(cfg);
          const etaSec = estimateConfigDurationSec(cfg);
          return {
            idx: idx + 1,
            code: sp.code,
            role: sp.role,
            reasonKey: sp.reasonKey,
            cfg,
            formSnap,
            advSnap,
            initCategorySnap: sp.nukeNative ? 'scratch' : initCategory.value,
            diff,
            etaSec,
          };
        });
      }

      // a face shot in the combination panel: find out whether RAFT can run here, so that its recipe can be offered
      watch(
        () => labComboCfg.panelOpen && effectiveLabComboTaskType.value === 'face_edit',
        (want) => {
          if (want && typeof ensureKfFlowCaps === 'function') ensureKfFlowCaps();
        },
        { immediate: true }
      );

      // Why the combination list has no keyframe-commonality recipe for a face / paint shot (null = nothing to say)
      const labComboKfNote = computed(() => {
        const effType = effectiveLabComboTaskType.value;
        if (effType !== 'face_edit' && effType !== 'cleanup_grain') return null;
        const ds = selectedDataset.value;
        const cIn = Number((ds && ds.in_channels) || form.in_channels || 3);
        const cOut = effType === 'face_edit' ? 3 : Number((ds && ds.out_channels) || form.out_channels || 3);
        const why = kfCommonBlocker(effType === 'face_edit' ? 3 : cIn, cOut, ds);
        return why ? t('lab.comboKfBlocked_' + why, { count: ds ? ds.frame_count : 0 }) : null;
      });

      const labComboSuitePreview = computed(() => {
        const ds = selectedDataset.value;
        const effType = effectiveLabComboTaskType.value;
        const candidates = _buildComboCandidateRecipes(ds, effType, labComboCfg.scope);
        const enabledGpus = (queue.gpus || []).filter((g) => g && g.enabled !== false);
        const gpuSlots = Math.max(
          1,
          (enabledGpus.length || 1) * Math.max(1, Number(adminSettings.max_jobs_per_gpu || 1))
        );
        // Greedy multi-GPU makespan estimate
        const bins = new Array(gpuSlots).fill(0);
        for (const c of candidates) {
          let minIdx = 0;
          for (let i = 1; i < bins.length; i++) {
            if (bins[i] < bins[minIdx]) minIdx = i;
          }
          bins[minIdx] += c.etaSec;
        }
        const maxBudgetParallelSec = bins.length ? Math.max(...bins) : 0;

        return {
          effType,
          gpuSlots,
          candidates,
          maxBudgetParallelSec,
          defaultGroupName: t('lab.comboDefaultGroupFmt', {
            type: t(`adv.preset_${effType}`),
            count: candidates.length,
          }),
        };
      });

      async function stageRecommendedComboSuite(launchImmediately = false) {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        const ds = selectedDataset.value;
        if (!ds || !(ds.source_path || ds.rel_path) || !(ds.target_path || ds.rel_path)) {
          notify(t('msg.errSelectDatasetFirst'), 'error');
          return;
        }
        if (initCategory.value === 'checkpoint' && !form.checkpoint_path) {
          notify(t('msg.errSelectCheckpointFirst'), 'error');
          return;
        }
        const suite = labComboSuitePreview.value;
        if (!suite.candidates.length) return;

        const taken = new Set([
          ...tasks.value.map((x) => x.node_name),
          ...labStaged.value.map((x) => x.node_name),
        ]);
        const allocateUniqueComboName = (idx, code) => {
          const cleanCode = String(code || 'Run')
            .replace(/[^A-Za-z0-9_-]/g, '')
            .slice(0, 16);
          let n = idx;
          let cand = `Combo${n}_${cleanCode}`;
          while (taken.has(cand)) {
            n += 1;
            cand = `Combo${n}_${cleanCode}`;
          }
          taken.add(cand);
          return cand;
        };

        const groupTitle = (labComboCfg.groupName || '').trim() || suite.defaultGroupName;
        let targetGrp = activeStagedGroup.value;
        if (targetGrp && targetGrp.items.length === 0 && !(targetGrp.name || '').trim()) {
          targetGrp.name = groupTitle;
          targetGrp.collapsed = false;
        } else {
          targetGrp = addNewLabStagedGroup(groupTitle);
          targetGrp.collapsed = false;
        }

        const stagedItems = suite.candidates.map((cand) => {
          const nodeName = allocateUniqueComboName(cand.idx, cand.code);
          const cfgCopy = JSON.parse(JSON.stringify(cand.cfg));
          cfgCopy.run_name = nodeName;
          return {
            uid: `st_${Date.now()}_${cand.idx}_${Math.random().toString(36).slice(2, 7)}`,
            node_name: nodeName,
            preferred_gpu: form.preferred_gpu || 'auto',
            dataset_name: ds.name || '',
            dataset_id: ds.id || null,
            note: t(cand.reasonKey),
            diff: summarizeClientConfig(cfgCopy),
            config: cfgCopy,
            formSnap: JSON.parse(JSON.stringify(cand.formSnap)),
            advSnap: JSON.parse(JSON.stringify(cand.advSnap)),
            initCategorySnap: cand.initCategorySnap,
          };
        });

        targetGrp.items.push(...stagedItems);
        labActiveStagedGroupId.value = targetGrp.gid;
        form.run_name = nextUniqueExpNodeName();

        if (launchImmediately) {
          await submitLabStagedGroup(targetGrp.gid);
        } else {
          notify(
            t('msg.labComboStagedToast', {
              count: stagedItems.length,
              group: groupTitle,
            }),
            'success'
          );
        }
      }

      function loadStagedToForm(item) {
        if (!item) return;
        if (item.dataset_id && pairedAssets.value.some((a) => a.id === item.dataset_id)) {
          selectedDatasetId.value = item.dataset_id;
        }
        if (item.formSnap) {
          Object.assign(form, JSON.parse(JSON.stringify(item.formSnap)));
          form.run_name = nextUniqueExpNodeName();
        }
        if (item.initCategorySnap) {
          initCategory.value = item.initCategorySnap;
        }
        syncLockedChannels(form.in_channels, form.out_channels);
        if (selectedDataset.value) {
          form.source = selectedDataset.value.source_path || selectedDataset.value.rel_path || form.source;
          form.target = selectedDataset.value.target_path || selectedDataset.value.rel_path || form.target;
          ensureAssetFrames(selectedDataset.value.id);
        }
        resetAdvToNukeNative(true);
        if (item.advSnap) {
          for (const k of ADV_KEYS) {
            if (item.advSnap[k]) Object.assign(adv[k], JSON.parse(JSON.stringify(item.advSnap[k])));
          }
        }
        if (!form.multi_resolution) adv.stage_schedule.enabled = false;
        if (!form.opt_fused_adam) form.opt_cuda_graph = false;
        if (Number(form.out_channels) === 1) {
          adv.loss_family.hdr_alpha = 0;
          if (adv.loss_family.enabled) syncLossFamilyEnabled();
        }
        if (Number(form.in_channels) < 3 && Number(form.out_channels) < 3) {
          adv.augment.exposure_ev = 0;
          if (adv.augment.enabled) syncAugmentEnabled();
        }
        syncKfCommonParams();
        labNote.value = item.note || '';
        notify(t('msg.labLoadedToFormToast'), 'info');
      }

      function removeLabStaged(uid) {
        for (const g of labStagedGroups.value) {
          if (Array.isArray(g.items)) {
            g.items = g.items.filter((x) => x.uid !== uid);
          }
        }
      }

      function clearLabStaged() {
        labStagedGroups.value = [createEmptyStagedGroup(1)];
        labActiveStagedGroupId.value = labStagedGroups.value[0].gid;
      }

      async function _submitSingleStagedGroupInternal(grp, fallbackIdx = 1) {
        if (!grp || !Array.isArray(grp.items) || !grp.items.length) return null;
        const itemsPayload = grp.items.map((it) => {
          const cfg = JSON.parse(JSON.stringify(it.config));
          cfg.run_name = (it.node_name || cfg.run_name || 'Exp_Run').trim();
          return {
            node_name: cfg.run_name,
            preferred_gpu: it.preferred_gpu || 'auto',
            priority: 0,
            label: it.diff && it.diff.tags && it.diff.tags.length ? it.diff.tags.join(' · ') : '',
            note: it.note || '',
            config: cfg,
          };
        });
        const targetBid = (grp.targetBatchId || '').trim() || null;
        let bName = (grp.name || '').trim();
        if (!bName && !targetBid) {
          bName = (labBatchName.value || '').trim();
        }
        const res = await apiFetch('/api/v1/experiments/batch-submit', {
          method: 'POST',
          body: JSON.stringify({
            batch_name: bName,
            target_batch_id: targetBid,
            eval_asset_id: labEvalAssetId.value || null,
            items: itemsPayload,
          }),
        });
        if (res.lab_groups) applyServerLabGroups(res.lab_groups);
        return res;
      }

      async function submitLabStagedGroup(gid) {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        const grp = labStagedGroups.value.find((g) => g.gid === gid) || activeStagedGroup.value;
        if (!grp || !grp.items || !grp.items.length || labSubmitting.value) return;
        labSubmitting.value = true;
        try {
          const res = await _submitSingleStagedGroupInternal(grp, 1);
          if (!res) return;
          labActiveBatchId.value = res.batch_id;
          if (labStagedGroups.value.length > 1) {
            removeLabStagedGroup(grp.gid);
          } else {
            grp.items = [];
            grp.name = '';
            grp.targetBatchId = '';
          }
          await Promise.all([fetchQueueAndTasks(), refreshMeQuietly()]);
          const newIds = (res.tasks || []).map((x) => x.id);
          if (newIds.length) {
            labSelectedTaskIds.value = newIds;
            const nukeRun = (res.tasks || []).find((x) => summarizeClientConfig(x.config).is_nuke_native);
            labBaselineTaskId.value = nukeRun ? nukeRun.id : newIds[0];
          }
          form.run_name = nextUniqueExpNodeName();
          notify(t('msg.labBatchSubmittedToast', { count: res.count }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          labSubmitting.value = false;
        }
      }

      async function submitAllLabStagedGroups() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        const ready = labStagedGroups.value.filter((g) => Array.isArray(g.items) && g.items.length > 0);
        if (!ready.length || labSubmitting.value) return;
        labSubmitting.value = true;
        let totalSubmitted = 0;
        let lastBatchId = '';
        let firstBatchTaskIds = [];
        let firstBatchTasks = [];
        try {
          for (let i = 0; i < ready.length; i++) {
            const grp = ready[i];
            const res = await _submitSingleStagedGroupInternal(grp, i + 1);
            if (res) {
              totalSubmitted += Number(res.count || 0);
              lastBatchId = res.batch_id;
              if (!firstBatchTaskIds.length && Array.isArray(res.tasks) && res.tasks.length) {
                firstBatchTasks = res.tasks;
                firstBatchTaskIds = res.tasks.map((x) => x.id);
              }
              grp.items = [];
            }
          }
          labStagedGroups.value = [createEmptyStagedGroup(1)];
          labActiveStagedGroupId.value = labStagedGroups.value[0].gid;
          if (lastBatchId) labActiveBatchId.value = lastBatchId;
          await Promise.all([fetchQueueAndTasks(), refreshMeQuietly()]);
          if (firstBatchTaskIds.length) {
            labSelectedTaskIds.value = firstBatchTaskIds;
            const nukeRun = firstBatchTasks.find((x) => summarizeClientConfig(x.config).is_nuke_native);
            labBaselineTaskId.value = nukeRun ? nukeRun.id : firstBatchTaskIds[0];
          }
          form.run_name = nextUniqueExpNodeName();
          notify(t('msg.labBatchSubmittedToast', { count: totalSubmitted }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          labSubmitting.value = false;
        }
      }

      async function submitLabBatch() {
        return submitLabStagedGroup(activeStagedGroup.value.gid);
      }

      async function submitLabDirectOrStaged() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        if (labSubmitting.value) return;
        if (labTotalStagedCount.value > 0) {
          if (labNonEmptyStagedGroupsCount.value > 1) {
            await submitAllLabStagedGroups();
          } else {
            const nonEmpty = labStagedGroups.value.find((g) => Array.isArray(g.items) && g.items.length > 0);
            await submitLabStagedGroup(nonEmpty ? nonEmpty.gid : activeStagedGroup.value.gid);
          }
          return;
        }
        if (!prepareFormForStaging()) return;
        const cfg = buildConfigPayload(false);
        const diff = summarizeClientConfig(cfg);
        const taken = new Set([
          ...tasks.value.map((x) => x.node_name),
          ...labStaged.value.map((x) => x.node_name),
        ]);
        let nodeName = (form.run_name || '').trim();
        if (!nodeName || taken.has(nodeName)) {
          nodeName = nextUniqueExpNodeName(diff.is_nuke_native ? 'NukeBase' : diff.model_size);
        }
        cfg.run_name = nodeName;
        const grp = activeStagedGroup.value;
        grp.collapsed = false;
        grp.items.push({
          uid: `st_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
          node_name: nodeName,
          preferred_gpu: form.preferred_gpu || 'auto',
          dataset_name: selectedDataset.value ? selectedDataset.value.name : '',
          dataset_id: selectedDataset.value ? selectedDataset.value.id : null,
          note: (labNote.value || '').trim(),
          diff,
          config: cfg,
          formSnap: JSON.parse(JSON.stringify(form)),
          advSnap: JSON.parse(JSON.stringify(adv)),
          initCategorySnap: initCategory.value,
        });
        labNote.value = '';
        await submitLabStagedGroup(grp.gid);
      }

      // ───────── Experiment Group Lists & Manual Adjustment (Analyze + Batch Monitor) ─────────
      const labNamedGroups = computed(() => {
        const map = new Map();
        for (const g of labGroupMeta.groups || []) {
          if (g && g.batch_id && g.batch_id !== UNGROUPED_LAB_KEY) {
            map.set(g.batch_id, {
              batch_id: g.batch_id,
              batch_name: g.batch_name || g.batch_id,
            });
          }
        }
        for (const tk of tasks.value) {
          const exp = (tk.config && tk.config.experiment) || {};
          const bid = String(tk.batch_id || exp.batch_id || '').trim();
          if (bid && bid !== UNGROUPED_LAB_KEY && !map.has(bid)) {
            map.set(bid, {
              batch_id: bid,
              batch_name: String(tk.batch_name || exp.batch_name || bid).trim() || bid,
            });
          }
        }
        const ordered = [];
        const seen = new Set();
        for (const k of labGroupMeta.order || []) {
          if (k !== UNGROUPED_LAB_KEY && map.has(k) && !seen.has(k)) {
            seen.add(k);
            ordered.push(map.get(k));
          }
        }
        for (const [k, v] of map.entries()) {
          if (!seen.has(k)) {
            seen.add(k);
            ordered.push(v);
          }
        }
        return ordered;
      });

      const labGroupedTasks = computed(() => {
        const q = (labTaskFilter.value || '').trim().toLowerCase();
        const smode = labTaskSortMode.value || 'manual';
        const buckets = new Map();
        for (const tk of tasks.value) {
          const diff = summarizeClientConfig(tk.config);
          const exp = (tk.config && tk.config.experiment) || {};
          const rawBid = String(tk.batch_id || exp.batch_id || '').trim();
          const key = rawBid && rawBid !== UNGROUPED_LAB_KEY ? rawBid : UNGROUPED_LAB_KEY;
          const sortOrder = Number(tk.sort_order || exp.sort_order || 0);
          if (!buckets.has(key)) buckets.set(key, []);
          buckets.get(key).push({ task: tk, diff, exp, sortOrder });
        }

        const cmpManual = (a, b) => {
          if (a.sortOrder > 0 && b.sortOrder > 0 && a.sortOrder !== b.sortOrder) {
            return a.sortOrder - b.sortOrder;
          }
          if (a.sortOrder > 0 && b.sortOrder <= 0) return -1;
          if (b.sortOrder > 0 && a.sortOrder <= 0) return 1;
          return a.task.id - b.task.id;
        };

        for (const arr of buckets.values()) {
          arr.sort((a, b) => {
            if (smode === 'psnr_desc') {
              const va = a.task.best_psnr_srgb;
              const vb = b.task.best_psnr_srgb;
              const ha = va != null && Number.isFinite(Number(va));
              const hb = vb != null && Number.isFinite(Number(vb));
              if (ha && hb && Number(va) !== Number(vb)) return Number(vb) - Number(va);
              if (ha && !hb) return -1;
              if (!ha && hb) return 1;
            } else if (smode === 'hfi_asc') {
              const va = a.task.best_seq_hfi;
              const vb = b.task.best_seq_hfi;
              const ha = va != null && Number.isFinite(Number(va));
              const hb = vb != null && Number.isFinite(Number(vb));
              if (ha && hb && Number(va) !== Number(vb)) return Number(va) - Number(vb);
              if (ha && !hb) return -1;
              if (!ha && hb) return 1;
            } else if (smode === 'val_asc') {
              const va = a.task.best_val_mse;
              const vb = b.task.best_val_mse;
              const ha = va != null && Number.isFinite(Number(va));
              const hb = vb != null && Number.isFinite(Number(vb));
              if (ha && hb && Number(va) !== Number(vb)) return Number(va) - Number(vb);
              if (ha && !hb) return -1;
              if (!ha && hb) return 1;
            }
            return cmpManual(a, b);
          });
        }

        const nameLookup = new Map();
        for (const ng of labNamedGroups.value) {
          nameLookup.set(ng.batch_id, ng.batch_name);
        }

        const allKeys = [];
        const seenKeys = new Set();
        for (const k of labGroupMeta.order || []) {
          if (k === UNGROUPED_LAB_KEY) {
            if (buckets.has(UNGROUPED_LAB_KEY) && !seenKeys.has(k)) {
              seenKeys.add(k);
              allKeys.push(k);
            }
          } else if ((nameLookup.has(k) || buckets.has(k)) && !seenKeys.has(k)) {
            seenKeys.add(k);
            allKeys.push(k);
          }
        }
        for (const ng of labNamedGroups.value) {
          if (!seenKeys.has(ng.batch_id)) {
            seenKeys.add(ng.batch_id);
            allKeys.push(ng.batch_id);
          }
        }
        if (buckets.has(UNGROUPED_LAB_KEY) && !seenKeys.has(UNGROUPED_LAB_KEY)) {
          seenKeys.add(UNGROUPED_LAB_KEY);
          allKeys.push(UNGROUPED_LAB_KEY);
        }

        const selSet = new Set(labSelectedTaskIds.value);
        const groupsOut = [];
        for (const key of allKeys) {
          const isUngrouped = key === UNGROUPED_LAB_KEY;
          const rawItems = buckets.get(key) || [];
          const grpName = isUngrouped
            ? t('lab.ungroupedTitle')
            : nameLookup.get(key) || (rawItems[0] && rawItems[0].exp.batch_name) || key;

          let visibleItems = rawItems;
          if (q) {
            const grpMatch = grpName.toLowerCase().includes(q);
            if (!grpMatch) {
              visibleItems = rawItems.filter(({ task, diff, exp }) => {
                const hay = [
                  task.node_name,
                  grpName,
                  exp.note || '',
                  diff.model_size || '',
                  ...(diff.tags || []),
                ]
                  .join(' ')
                  .toLowerCase();
                return hay.includes(q);
              });
              if (!visibleItems.length) continue;
            }
          }

          const selectedCount = visibleItems.reduce((n, e) => n + (selSet.has(e.task.id) ? 1 : 0), 0);
          const doneCount = rawItems.reduce((n, e) => n + (e.task.status === 'completed' ? 1 : 0), 0);
          groupsOut.push({
            key,
            batch_id: isUngrouped ? '' : key,
            name: grpName,
            isUngrouped,
            items: visibleItems,
            allItems: rawItems,
            selectedCount,
            doneCount,
          });
        }
        return groupsOut;
      });

      const labFilteredTasks = computed(() => {
        const flat = [];
        for (const g of labGroupedTasks.value) {
          flat.push(...g.items);
        }
        return flat;
      });

      const labBatchGroups = computed(() => {
        return labGroupedTasks.value.filter((g) => !g.isUngrouped && g.allItems.length > 0);
      });

      const labBatchTasks = computed(() => {
        if (!labBatchGroups.value.length) return [];
        let grp = labBatchGroups.value.find((g) => g.batch_id === labActiveBatchId.value);
        if (!grp) grp = labBatchGroups.value[0];
        return grp.allItems.map((e) => e.task);
      });

      const labBatchStats = computed(() => {
        const list = labBatchTasks.value;
        const done = list.filter((x) => x.status === 'completed').length;
        return { total: list.length, done };
      });

      function openBatchInAnalyze() {
        const ids = labBatchTasks.value.map((x) => x.id);
        if (ids.length) {
          labSelectedTaskIds.value = ids;
          const nukeRun = labBatchTasks.value.find((x) => summarizeClientConfig(x.config).is_nuke_native);
          labBaselineTaskId.value = nukeRun ? nukeRun.id : ids[0];
        }
        labMode.value = 'analyze';
        fetchLabCompare();
      }

      function isLabGroupCollapsed(key) {
        return !!labCollapsedGroups[key];
      }

      function toggleLabGroupCollapse(key) {
        labCollapsedGroups[key] = !labCollapsedGroups[key];
      }

      const areAllLabGroupsCollapsed = computed(() => {
        if (!labGroupedTasks.value.length) return false;
        return labGroupedTasks.value.every((g) => !!labCollapsedGroups[g.key]);
      });

      function toggleAllLabGroupsCollapse() {
        const target = !areAllLabGroupsCollapsed.value;
        for (const g of labGroupedTasks.value) {
          labCollapsedGroups[g.key] = target;
        }
      }

      function isLabGroupAllSelected(grp) {
        if (!grp || !grp.items || !grp.items.length) return false;
        return grp.items.every((e) => labSelectedTaskIds.value.includes(e.task.id));
      }

      function selectOnlyLabGroup(grp) {
        if (!grp || !grp.items || !grp.items.length) return;
        labCollapsedGroups[grp.key] = false;
        const ids = grp.items.map((e) => e.task.id);
        labSelectedTaskIds.value = ids;
        const nukeEntry = grp.items.find((e) => e.diff && e.diff.is_nuke_native);
        labBaselineTaskId.value = nukeEntry ? nukeEntry.task.id : ids[0];
      }

      function toggleLabGroupSelection(grp) {
        if (!grp || !grp.items || !grp.items.length) return;
        const ids = grp.items.map((e) => e.task.id);
        if (isLabGroupAllSelected(grp)) {
          const removeSet = new Set(ids);
          labSelectedTaskIds.value = labSelectedTaskIds.value.filter((id) => !removeSet.has(id));
          if (removeSet.has(labBaselineTaskId.value)) {
            labBaselineTaskId.value = labSelectedTaskIds.value[0] || null;
          }
        } else {
          const curSet = new Set(labSelectedTaskIds.value);
          for (const id of ids) {
            if (!curSet.has(id)) {
              curSet.add(id);
              labSelectedTaskIds.value.push(id);
            }
          }
          if (!labBaselineTaskId.value) {
            const nukeEntry = grp.items.find((e) => e.diff && e.diff.is_nuke_native);
            labBaselineTaskId.value = nukeEntry ? nukeEntry.task.id : ids[0];
          }
        }
      }

      function openCreateLabGroup() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        labGroupCreateOpen.value = !labGroupCreateOpen.value;
        if (labGroupCreateOpen.value && !labNewGroupName.value) {
          labNewGroupName.value = '';
        }
      }

      async function confirmCreateLabGroup() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        const clean = (labNewGroupName.value || '').trim();
        if (!clean) return;
        const moveIds =
          labCreateMoveSelected.value && labSelectedTaskIds.value.length
            ? [...labSelectedTaskIds.value]
            : [];
        try {
          const res = await apiFetch('/api/v1/experiments/groups/create', {
            method: 'POST',
            body: JSON.stringify({
              batch_name: clean,
              task_ids: moveIds,
            }),
          });
          if (res.lab_groups) applyServerLabGroups(res.lab_groups);
          if (res.batch_id) labCollapsedGroups[res.batch_id] = false;
          labGroupCreateOpen.value = false;
          labNewGroupName.value = '';
          await fetchQueueAndTasks();
          notify(t('msg.labGroupCreatedToast', { name: res.batch_name || clean }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      function startRenameLabGroup(grp) {
        if (!grp || grp.isUngrouped) return;
        labRenamingGroupId.value = grp.batch_id;
        labRenamingGroupName.value = grp.name;
      }

      function cancelRenameLabGroup() {
        labRenamingGroupId.value = '';
        labRenamingGroupName.value = '';
      }

      async function confirmRenameLabGroup(grp) {
        if (!grp || grp.isUngrouped) return;
        const clean = (labRenamingGroupName.value || '').trim();
        if (!clean || clean === grp.name) {
          cancelRenameLabGroup();
          return;
        }
        try {
          const res = await apiFetch('/api/v1/experiments/groups/rename', {
            method: 'PUT',
            body: JSON.stringify({
              batch_id: grp.batch_id,
              batch_name: clean,
            }),
          });
          if (res.lab_groups) applyServerLabGroups(res.lab_groups);
          cancelRenameLabGroup();
          await fetchQueueAndTasks();
          notify(t('msg.labGroupRenamedToast', { name: clean }), 'success');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function moveLabGroupOrder(gIdx, dir) {
        const list = labGroupedTasks.value;
        const target = gIdx + dir;
        if (target < 0 || target >= list.length) return;
        const keys = list.map((g) => g.key);
        const tmp = keys[gIdx];
        keys[gIdx] = keys[target];
        keys[target] = tmp;
        labGroupMeta.order = [...keys];
        try {
          const res = await apiFetch('/api/v1/experiments/groups/reorder', {
            method: 'PUT',
            body: JSON.stringify({ group_order: keys }),
          });
          if (res.lab_groups) applyServerLabGroups(res.lab_groups);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function moveTaskWithinLabGroup(grp, tIdx, dir) {
        if (!grp || !grp.allItems || grp.allItems.length < 2) return;
        const curTaskId = grp.items[tIdx] && grp.items[tIdx].task.id;
        const neighborTaskId = grp.items[tIdx + dir] && grp.items[tIdx + dir].task.id;
        if (!curTaskId || !neighborTaskId) return;
        const fullIds = grp.allItems.map((e) => e.task.id);
        const aIdx = fullIds.indexOf(curTaskId);
        const bIdx = fullIds.indexOf(neighborTaskId);
        if (aIdx < 0 || bIdx < 0) return;
        fullIds[aIdx] = neighborTaskId;
        fullIds[bIdx] = curTaskId;
        // Optimistic local update for 0ms visual feedback
        fullIds.forEach((tid, pos) => {
          const tk = tasks.value.find((x) => x.id === tid);
          if (tk) {
            tk.sort_order = pos + 1;
            tk.config = tk.config || {};
            tk.config.experiment = tk.config.experiment || {};
            tk.config.experiment.sort_order = pos + 1;
          }
        });
        try {
          const res = await apiFetch('/api/v1/experiments/groups/reorder', {
            method: 'PUT',
            body: JSON.stringify({ task_ids: fullIds }),
          });
          if (res.lab_groups) applyServerLabGroups(res.lab_groups);
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function archiveSelectedLabTaskFiles() {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        if (labArchiving.value) return;
        const eligibleIds = labSelectedTaskIds.value.filter((tid) => {
          const tk = tasks.value.find((x) => x.id === tid);
          return tk && tk.status !== 'running' && tk.status !== 'queued';
        });
        if (!eligibleIds.length) {
          notify(t('msg.labSelectAtLeastOne'), 'warning');
          return;
        }
        if (!confirm(t('msg.confirmArchiveLabTasks', { count: eligibleIds.length }))) return;
        labArchiving.value = true;
        try {
          const res = await apiFetch('/api/v1/experiments/archive-files', {
            method: 'POST',
            body: JSON.stringify({
              task_ids: eligibleIds,
              keep_final_model: true,
            }),
          });
          if (res.db_summary) Object.assign(labDbSummary, res.db_summary);
          await Promise.all([fetchQueueAndTasks(), fetchCheckpoints()]);
          if (topNav.value === 'lab' && labMode.value === 'analyze') {
            await fetchLabCompare(true);
          }
          notify(
            t('msg.labTasksArchivedToast', {
              count: Number(res.archived_count || eligibleIds.length),
              freed: Number(res.freed_mb || 0).toFixed(1),
            }),
            'success'
          );
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          labArchiving.value = false;
        }
      }

      async function moveTasksToLabGroup(taskIds, rawTarget, selectEl = null) {
        if (selectEl) selectEl.value = '';
        if (!rawTarget || !Array.isArray(taskIds) || !taskIds.length) return;
        let targetBatchId = rawTarget;
        let targetBatchName = '';
        if (rawTarget === '__new__') {
          const nm = (typeof window !== 'undefined' ? window.prompt(t('msg.promptNewGroupForMove')) : '') || '';
          if (!nm.trim()) return;
          targetBatchName = nm.trim();
        }
        try {
          const res = await apiFetch('/api/v1/experiments/groups/move-tasks', {
            method: 'POST',
            body: JSON.stringify({
              task_ids: taskIds,
              target_batch_id: targetBatchId,
              target_batch_name: targetBatchName,
            }),
          });
          if (res.lab_groups) applyServerLabGroups(res.lab_groups);
          if (res.batch_id) labCollapsedGroups[res.batch_id] = false;
          await fetchQueueAndTasks();
          if (!res.batch_id) {
            notify(t('msg.labTasksUngroupedToast', { count: res.moved_count || taskIds.length }), 'success');
          } else {
            notify(
              t('msg.labTasksMovedToast', {
                count: res.moved_count || taskIds.length,
                name: res.batch_name || targetBatchName || res.batch_id,
              }),
              'success'
            );
          }
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      function onSingleTaskMoveSelect(taskId, evt) {
        const el = evt && evt.target;
        const val = el ? el.value : '';
        moveTasksToLabGroup([taskId], val, el);
      }

      function onBatchTaskMoveSelect(evt) {
        const el = evt && evt.target;
        const val = el ? el.value : '';
        moveTasksToLabGroup([...labSelectedTaskIds.value], val, el);
      }

      async function dissolveLabGroup(grp) {
        if (!grp || grp.isUngrouped || !grp.batch_id) return;
        if (grp.allItems && grp.allItems.length > 0) {
          const ok = confirm(
            t('msg.confirmDissolveLabGroup', { name: grp.name, count: grp.allItems.length })
          );
          if (!ok) return;
        }
        try {
          const res = await apiFetch('/api/v1/experiments/groups/dissolve', {
            method: 'POST',
            body: JSON.stringify({ batch_id: grp.batch_id }),
          });
          if (res.lab_groups) applyServerLabGroups(res.lab_groups);
          delete labCollapsedGroups[grp.batch_id];
          await fetchQueueAndTasks();
          notify(t('msg.labGroupDissolvedToast', { name: grp.name }), 'info');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      function labRunColor(taskId) {
        const idx = labSelectedTaskIds.value.indexOf(taskId);
        return LAB_COLORS[(idx >= 0 ? idx : 0) % LAB_COLORS.length];
      }

      function toggleLabTaskSelection(taskId) {
        const idx = labSelectedTaskIds.value.indexOf(taskId);
        if (idx >= 0) {
          labSelectedTaskIds.value.splice(idx, 1);
          if (labBaselineTaskId.value === taskId) {
            labBaselineTaskId.value = labSelectedTaskIds.value[0] || null;
          }
        } else {
          labSelectedTaskIds.value.push(taskId);
          if (!labBaselineTaskId.value) {
            labBaselineTaskId.value = taskId;
          }
        }
      }

      function selectFirstReadyLabGroup() {
        const readyGrp =
          labGroupedTasks.value.find(
            (g) => g.items.length > 0 && g.items.some((e) => e.task.status === 'completed' || e.task.last_step > 0)
          ) || labGroupedTasks.value.find((g) => g.items.length > 0);
        if (readyGrp) {
          selectOnlyLabGroup(readyGrp);
        }
      }

      function selectAllCompletedForLab() {
        const done = tasks.value.filter((x) => x.status === 'completed' || x.last_step > 0).slice(0, 8);
        labSelectedTaskIds.value = done.map((x) => x.id);
        const nukeRun = done.find((x) => summarizeClientConfig(x.config).is_nuke_native);
        labBaselineTaskId.value = nukeRun ? nukeRun.id : (labSelectedTaskIds.value[0] || null);
      }

      function clearLabSelection() {
        labSelectedTaskIds.value = [];
        labBaselineTaskId.value = null;
        labCompare.runs = [];
      }

      let _labCompareReqSeq = 0;

      async function fetchLabCompare(silent = false) {
        if (!user.value || !canUseLab.value) return;
        const reqSeq = ++_labCompareReqSeq;
        if (!labSelectedTaskIds.value.length) {
          labCompare.runs = [];
          labCompare.loading = false;
          await nextTick();
          drawLabCompareChart();
          return;
        }
        if (!silent) labCompare.loading = true;
        try {
          const res = await apiFetch('/api/v1/experiments/compare', {
            method: 'POST',
            body: JSON.stringify({
              task_ids: labSelectedTaskIds.value,
              eval_asset_id: labEvalAssetId.value || null,
              ckpt_mode: labCkptMode.value || 'best',
            }),
          });
          if (reqSeq !== _labCompareReqSeq) return;
          labCompare.runs = res.runs || [];
          labCompare.eval_asset = res.eval_asset || null;
          if (res.eval_runner) Object.assign(labCompare.eval_runner, res.eval_runner);
          if (
            labCompare.runs.length &&
            (!labBaselineTaskId.value || !labCompare.runs.some((r) => r.task_id === labBaselineTaskId.value))
          ) {
            const nukeRun = labCompare.runs.find((r) => r.param_diff && r.param_diff.is_nuke_native);
            labBaselineTaskId.value = nukeRun ? nukeRun.task_id : labCompare.runs[0].task_id;
          }
          const maxF = labSeqFrameCount.value;
          if (maxF > 0 && labViewer.index >= maxF) {
            labViewer.index = 0;
          }
          await nextTick();
          drawLabCompareChart();
        } catch (e) {
          if (!silent && reqSeq === _labCompareReqSeq) notify(e.message, 'error');
        } finally {
          if (reqSeq === _labCompareReqSeq) labCompare.loading = false;
        }
      }

      async function triggerLabEvaluation(force = false) {
        if (!canUseLab.value) {
          notify(t('server.tempUserLabLocked'), 'warning');
          return;
        }
        const validIds = labCompare.runs.filter((r) => r.has_ckpt).map((r) => r.task_id);
        const ids = validIds.length ? validIds : labSelectedTaskIds.value;
        if (!ids.length) {
          notify(t('msg.labSelectAtLeastOne'), 'warning');
          return;
        }
        try {
          const res = await apiFetch('/api/v1/experiments/evaluate', {
            method: 'POST',
            body: JSON.stringify({
              task_ids: ids,
              eval_asset_id: labEvalAssetId.value || null,
              ckpt_mode: labCkptMode.value || 'best',
              force: !!force,
            }),
          });
          if (res.eval_runner) Object.assign(labCompare.eval_runner, res.eval_runner);
          if (res.eval_runner && res.eval_runner.added > 0) {
            notify(t('msg.labEvalStartedToast', { count: res.eval_runner.added }), 'success');
          } else {
            notify(t('msg.labEvalAllCachedToast'), 'info');
            await fetchLabCompare(true);
          }
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      async function cancelLabEvaluation() {
        try {
          const res = await apiFetch('/api/v1/experiments/eval-cancel', { method: 'POST' });
          if (res.eval_runner) Object.assign(labCompare.eval_runner, res.eval_runner);
          notify(t('msg.labEvalCancelledToast'), 'warning');
        } catch (e) {
          notify(e.message, 'error');
        }
      }

      // ───────── Lab Multi-Series Curve Renderer (Loss / Val MSE / Per-Frame Flicker) ─────────
      const activeLabDataset = computed(() => {
        if (labEvalAssetId.value) {
          const explicit = assets.value.find((a) => a.id === labEvalAssetId.value);
          if (explicit) return explicit;
        }
        if (labCompare.eval_asset && labCompare.eval_asset.id) {
          const matched = assets.value.find((a) => a.id === labCompare.eval_asset.id);
          return matched || labCompare.eval_asset;
        }
        for (const tid of labSelectedTaskIds.value) {
          const tk = tasks.value.find((x) => x.id === tid);
          if (!tk || !tk.config) continue;
          const did = tk.config.experiment && tk.config.experiment.dataset_id;
          if (did) {
            const m = assets.value.find((a) => a.id === Number(did));
            if (m) return m;
          }
          const sp = String((tk.config.paths && tk.config.paths.source) || '').replace(/\\/g, '/');
          if (sp) {
            const m = assets.value.find(
              (a) =>
                String(a.source_path || '').replace(/\\/g, '/') === sp ||
                String(a.rel_path || '').replace(/\\/g, '/') === sp
            );
            if (m) return m;
          }
        }
        return null;
      });

      const hasLabEvalSequence = computed(() => {
        if (labCompare.eval_asset && labCompare.eval_asset.id) {
          return !!labCompare.eval_asset.has_eval_sequence;
        }
        const ds = activeLabDataset.value;
        if (!ds) return false;
        if (ds.kind === 'eval_sequence') return Number(ds.frame_count || 0) >= 2;
        return !!ds.has_eval_sequence || Number(ds.seq_frame_count || 0) >= 2;
      });

      const isLabPerFrameChart = computed(() => {
        return ['seq_hfi', 'seq_delta', 'seq_psnr'].includes(labChartTab.value);
      });

      watch(hasLabEvalSequence, (hasSeq) => {
        if (!hasSeq && isLabPerFrameChart.value) {
          labChartTab.value = 'train_loss';
          labViewer.playing = false;
        }
      });

      function extractLabSeries(run, tab) {
        if (tab === 'train_loss') {
          const raw = (run.steps || []).filter((s) => Number.isFinite(s.total) && s.total >= 0);
          if (!raw.length) return [];
          const w = Number(labChartCfg.smooth || 0);
          let last = raw[0].total;
          return raw.map((p) => {
            last = last * w + p.total * (1 - w);
            return { x: p.step, y: w > 0 ? last : p.total, rawY: p.total };
          });
        }
        if (tab === 'val_mse') {
          return (run.vals || [])
            .filter((v) => Number.isFinite(v.mse) && v.mse >= 0)
            .map((v) => ({ x: v.step, y: v.mse, rawY: v.mse }));
        }
        const pf = (run.eval && run.eval.sequence && run.eval.sequence.per_frame) || [];
        if (tab === 'seq_hfi') {
          return pf
            .filter((r) => r.flicker_hf != null && Number.isFinite(r.flicker_hf))
            .map((r) => ({ x: r.idx, y: r.flicker_hf * 1000, rawY: r.flicker_hf * 1000 }));
        }
        if (tab === 'seq_delta') {
          return pf
            .filter((r) => r.flicker_delta != null && Number.isFinite(r.flicker_delta))
            .map((r) => ({ x: r.idx, y: r.flicker_delta * 1000, rawY: r.flicker_delta * 1000 }));
        }
        if (tab === 'seq_psnr') {
          return pf
            .filter((r) => r.psnr_srgb != null && Number.isFinite(r.psnr_srgb))
            .map((r) => ({ x: r.idx, y: r.psnr_srgb, rawY: r.psnr_srgb }));
        }
        return [];
      }

      function drawLabCompareChart() {
        const canvas = labCompareCanvas.value;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
        const W = Math.max(320, rect.width || 640);
        const H = Math.max(220, rect.height || 260);
        canvas.width = W * dpr;
        canvas.height = H * dpr;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, W, H);

        const tab = labChartTab.value;
        const useLog = labChartCfg.logScale && tab !== 'seq_psnr' && !isLabPerFrameChart.value;
        const tf = (v) => (useLog ? Math.log10(Math.max(1e-10, v)) : v);

        const seriesList = labCompare.runs
          .map((run) => ({
            task_id: run.task_id,
            name: run.node_name,
            color: labRunColor(run.task_id),
            pts: extractLabSeries(run, tab),
          }))
          .filter((s) => s.pts.length > 0);

        if (!seriesList.length) {
          labChartPlotMeta = null;
          ctx.fillStyle = '#64748b';
          ctx.font = '12px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(t('lab.chartNoData'), W / 2, H / 2);
          return;
        }

        const padL = 66, padR = 24, padT = 18, padB = 30;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const s of seriesList) {
          for (const p of s.pts) {
            const ty = tf(p.y);
            if (p.x < minX) minX = p.x;
            if (p.x > maxX) maxX = p.x;
            if (ty < minY) minY = ty;
            if (ty > maxY) maxY = ty;
          }
        }
        if (minX === maxX) maxX = minX + 1;
        if (minY === maxY) {
          minY -= 0.5;
          maxY += 0.5;
        } else {
          const margin = (maxY - minY) * 0.06;
          minY -= margin;
          maxY += margin;
        }

        const toX = (x) => padL + ((x - minX) / (maxX - minX)) * plotW;
        const toY = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;
        labChartPlotMeta = { padL, padR, padT, padB, plotW, plotH, minX, maxX, seriesList, isFrame: isLabPerFrameChart.value };

        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.fillStyle = '#64748b';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
          const gy = padT + (i / 5) * plotH;
          const val = maxY - (i / 5) * (maxY - minY);
          let disp;
          if (useLog) disp = Math.pow(10, val).toExponential(1);
          else if (tab === 'seq_psnr') disp = val.toFixed(2) + ' dB';
          else if (isLabPerFrameChart.value) disp = val.toFixed(2) + '‰';
          else disp = val.toFixed(5);
          ctx.beginPath();
          ctx.moveTo(padL, gy);
          ctx.lineTo(W - padR, gy);
          ctx.stroke();
          ctx.fillText(disp, padL - 8, gy + 4);
        }

        ctx.textAlign = 'center';
        const xTicks = 5;
        for (let i = 0; i <= xTicks; i++) {
          const xVal = Math.round(minX + (i / xTicks) * (maxX - minX));
          const gx = padL + (i / xTicks) * plotW;
          ctx.strokeStyle = 'rgba(30, 41, 59, 0.55)';
          ctx.beginPath();
          ctx.moveTo(gx, padT);
          ctx.lineTo(gx, padT + plotH);
          ctx.stroke();
          ctx.fillStyle = '#64748b';
          ctx.fillText(isLabPerFrameChart.value ? `#${xVal + 1}` : `Step ${xVal}`, gx, H - 8);
        }

        // Active frame marker on per-frame sequence charts
        if (isLabPerFrameChart.value && labViewer.index >= minX && labViewer.index <= maxX) {
          const fx = toX(labViewer.index);
          ctx.save();
          ctx.setLineDash([3, 3]);
          ctx.strokeStyle = 'rgba(251, 191, 36, 0.75)';
          ctx.beginPath();
          ctx.moveTo(fx, padT);
          ctx.lineTo(fx, padT + plotH);
          ctx.stroke();
          ctx.restore();
        }

        for (const s of seriesList) {
          ctx.strokeStyle = s.color;
          ctx.fillStyle = s.color;
          ctx.lineWidth = s.task_id === labBaselineTaskId.value ? 2.4 : 1.8;
          ctx.beginPath();
          s.pts.forEach((p, idx) => {
            const cx = toX(p.x);
            const cy = toY(tf(p.y));
            if (idx === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
          });
          ctx.stroke();
          if (s.pts.length <= 80) {
            s.pts.forEach((p) => {
              const cx = toX(p.x);
              const cy = toY(tf(p.y));
              ctx.beginPath();
              ctx.arc(cx, cy, 2.5, 0, Math.PI * 2);
              ctx.fill();
            });
          }
        }

        if (labChartHover.value && labChartHover.value.x >= minX && labChartHover.value.x <= maxX) {
          const hx = toX(labChartHover.value.x);
          ctx.save();
          ctx.strokeStyle = 'rgba(148, 163, 184, 0.55)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(hx, padT);
          ctx.lineTo(hx, padT + plotH);
          ctx.stroke();
          ctx.restore();
        }
      }

      function onLabChartMouseMove(e) {
        if (!labChartPlotMeta || !labCompareCanvas.value) return;
        const rect = labCompareCanvas.value.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const { padL, plotW, minX, maxX, seriesList, isFrame } = labChartPlotMeta;
        const clamped = Math.max(0, Math.min(1, (mx - padL) / Math.max(1, plotW)));
        const targetX = Math.round(minX + clamped * (maxX - minX));
        const values = [];
        for (const s of seriesList) {
          let best = s.pts[0];
          let bestDist = Math.abs(best.x - targetX);
          for (const p of s.pts) {
            const d = Math.abs(p.x - targetX);
            if (d < bestDist) {
              best = p;
              bestDist = d;
            }
          }
          values.push({
            task_id: s.task_id,
            name: s.name,
            color: s.color,
            val: best.y,
          });
        }
        labChartHover.value = { x: targetX, isFrame, values };
        drawLabCompareChart();
      }

      function onLabChartMouseLeave() {
        labChartHover.value = null;
        drawLabCompareChart();
      }

      function onLabChartClick(e) {
        if (!labChartPlotMeta || !labChartPlotMeta.isFrame || !labCompareCanvas.value) return;
        const rect = labCompareCanvas.value.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const { padL, plotW, minX, maxX } = labChartPlotMeta;
        const clamped = Math.max(0, Math.min(1, (mx - padL) / Math.max(1, plotW)));
        const targetIdx = Math.round(minX + clamped * (maxX - minX));
        if (labSeqFrameCount.value > 0) {
          labViewer.index = Math.max(0, Math.min(labSeqFrameCount.value - 1, targetIdx));
          drawLabCompareChart();
        }
      }

      // ───────── Quantitative Metrics, Baseline Δ & Synchronized Sequence Viewer ─────────
      const labBaselineRun = computed(() => {
        if (!labCompare.runs.length) return null;
        return (
          labCompare.runs.find((r) => r.task_id === labBaselineTaskId.value) || labCompare.runs[0]
        );
      });

      function getRunMetric(run, key) {
        if (!run) return null;
        const sup = (run.eval && run.eval.supervised) || {};
        const seq = (run.eval && run.eval.sequence) || {};
        if (key === 'best_val') return run.best_val_mse;
        if (key === 'sup_psnr') return sup.psnr_srgb;
        if (key === 'sup_edge') return sup.edge_l1 != null ? sup.edge_l1 * 1000 : null;
        if (key === 'sup_bg') return sup.bg_drift_mae != null ? sup.bg_drift_mae * 1000 : null;
        if (key === 'sup_p99') return sup.p99_err != null ? sup.p99_err * 1000 : null;
        if (key === 'seq_delta') return seq.flicker_delta != null ? seq.flicker_delta * 1000 : null;
        if (key === 'seq_hfi') return seq.flicker_hf != null ? seq.flicker_hf * 1000 : null;
        if (key === 'seq_static') return seq.flicker_static != null ? seq.flicker_static * 1000 : null;
        return null;
      }

      function isBestLabMetric(run, key, higherBetter = false) {
        const v = getRunMetric(run, key);
        if (v == null || !Number.isFinite(v) || labCompare.runs.length < 2) return false;
        const vals = labCompare.runs
          .map((r) => getRunMetric(r, key))
          .filter((x) => x != null && Number.isFinite(x));
        if (vals.length < 2) return false;
        const target = higherBetter ? Math.max(...vals) : Math.min(...vals);
        return Math.abs(v - target) <= 1e-12;
      }

      function formatLabDelta(run, key, higherBetter = false, digits = 2) {
        const base = labBaselineRun.value;
        if (!base || !run || run.task_id === base.task_id) return null;
        const v = getRunMetric(run, key);
        const bv = getRunMetric(base, key);
        if (v == null || bv == null || !Number.isFinite(v) || !Number.isFinite(bv)) return null;
        const diff = v - bv;
        if (Math.abs(diff) < 1e-9) return { text: '0.00', cls: 'text-slate-500' };
        const sign = diff > 0 ? '+' : '';
        const good = higherBetter ? diff > 0 : diff < 0;
        const txt = key === 'best_val' ? `${sign}${diff.toExponential(1)}` : `${sign}${diff.toFixed(digits)}`;
        return {
          text: txt,
          cls: good ? 'text-emerald-400' : 'text-rose-400',
        };
      }

      const labEvaluatedSeqRuns = computed(() => {
        return labCompare.runs.filter(
          (r) => r.eval && r.eval.sequence && Array.isArray(r.eval.sequence.per_frame) && r.eval.sequence.per_frame.length > 0
        );
      });

      const labSeqHasPairedErr = computed(() => {
        return labEvaluatedSeqRuns.value.some((r) => r.eval && r.eval.sequence && r.eval.sequence.is_paired);
      });
      // per-frame PSNR needs per-frame ground truth in the validation sequence (paired); without it the tab is not offered
      watch(
        () => labSeqHasPairedErr.value,
        (paired) => {
          if (!paired && labChartTab.value === 'seq_psnr') labChartTab.value = 'train_loss';
        }
      );

      const labSeqFrameCount = computed(() => {
        if (labEvaluatedSeqRuns.value.length) {
          return labEvaluatedSeqRuns.value[0].eval.sequence.per_frame.length;
        }
        if (labCompare.eval_asset && Array.isArray(labCompare.eval_asset.source_frames)) {
          return labCompare.eval_asset.source_frames.length;
        }
        return 0;
      });

      const labGlobalPeakIdx = computed(() => {
        let bestIdx = 0;
        let bestVal = -1;
        for (const r of labEvaluatedSeqRuns.value) {
          const seq = r.eval.sequence;
          if (seq && seq.peak_flicker_val != null && seq.peak_flicker_val > bestVal) {
            bestVal = seq.peak_flicker_val;
            bestIdx = Number(seq.peak_flicker_idx || 0);
          }
        }
        return bestIdx;
      });

      const labCurrentSourceFrame = computed(() => {
        const sf = (labCompare.eval_asset && labCompare.eval_asset.source_frames) || [];
        if (!sf.length) return null;
        const idx = Math.min(Math.max(0, labViewer.index), sf.length - 1);
        return sf[idx];
      });

      function getLabRunFrame(run) {
        const pf = (run && run.eval && run.eval.sequence && run.eval.sequence.per_frame) || [];
        if (!pf.length) return null;
        const idx = Math.min(Math.max(0, labViewer.index), pf.length - 1);
        const row = pf[idx];
        let url = row.out_url;
        if (labViewer.mode === 'flicker') url = row.flicker_url || row.out_url;
        else if (labViewer.mode === 'err') url = row.err_url || row.out_url;
        return { ...row, active_url: url };
      }

      function jumpLabToFrame(idx) {
        const tot = labSeqFrameCount.value;
        if (tot <= 0) return;
        labViewer.playing = false;
        labViewer.index = Math.max(0, Math.min(tot - 1, Number(idx || 0)));
        if (isLabPerFrameChart.value) drawLabCompareChart();
      }

      function stepLabFrame(delta) {
        const tot = labSeqFrameCount.value;
        if (tot <= 0) return;
        labViewer.index = (labViewer.index + delta + tot) % tot;
        if (isLabPerFrameChart.value) drawLabCompareChart();
      }

      function resetLabZoom() {
        labViewer.zoom = 1;
        labViewer.panX = 0;
        labViewer.panY = 0;
      }

      function onLabWheel(e) {
        const factor = e.deltaY < 0 ? 1.14 : 0.88;
        labViewer.zoom = Math.min(12, Math.max(0.35, labViewer.zoom * factor));
      }

      function onLabMouseDown(e) {
        labViewer.dragging = true;
        labViewer.dragStartX = e.clientX - labViewer.panX;
        labViewer.dragStartY = e.clientY - labViewer.panY;
      }

      function onLabMouseMove(e) {
        if (!labViewer.dragging) return;
        labViewer.panX = e.clientX - labViewer.dragStartX;
        labViewer.panY = e.clientY - labViewer.dragStartY;
      }

      function onLabMouseUp() {
        labViewer.dragging = false;
      }

      function exportLabCsv() {
        if (!labCompare.runs.length) return;
        const headers = [
          'task_id',
          'node_name',
          'is_baseline',
          'ckpt_step',
          'params',
          'best_val_mse',
          'keyframe_psnr_srgb_db',
          'keyframe_psnr_aces_db',
          'keyframe_edge_l1_x1000',
          'keyframe_bg_drift_x1000',
          'keyframe_p99_err_x1000',
          'seq_asset',
          'seq_frames',
          'seq_flicker_delta_permille',
          'seq_flicker_hfi_permille',
          'seq_static_boil_permille',
          'seq_peak_flicker_frame',
        ];
        const esc = (v) => {
          if (v == null) return '';
          const s = String(v);
          return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
        };
        const lines = [headers.join(',')];
        for (const r of labCompare.runs) {
          const sup = (r.eval && r.eval.supervised) || {};
          const seq = (r.eval && r.eval.sequence) || {};
          const ptags =
            r.param_diff && r.param_diff.is_nuke_native
              ? 'NukeNative'
              : ((r.param_diff && r.param_diff.tags) || []).join(' | ');
          lines.push(
            [
              r.task_id,
              r.node_name,
              r.task_id === (labBaselineRun.value && labBaselineRun.value.task_id) ? '1' : '0',
              r.ckpt_step || 0,
              ptags,
              r.best_val_mse != null ? r.best_val_mse : '',
              sup.psnr_srgb != null ? sup.psnr_srgb.toFixed(4) : '',
              sup.psnr_aces != null ? sup.psnr_aces.toFixed(4) : '',
              sup.edge_l1 != null ? (sup.edge_l1 * 1000).toFixed(4) : '',
              sup.bg_drift_mae != null ? (sup.bg_drift_mae * 1000).toFixed(4) : '',
              sup.p99_err != null ? (sup.p99_err * 1000).toFixed(4) : '',
              (r.eval && r.eval.eval_asset_name) || '',
              seq.frame_count || '',
              seq.flicker_delta != null ? (seq.flicker_delta * 1000).toFixed(4) : '',
              seq.flicker_hf != null ? (seq.flicker_hf * 1000).toFixed(4) : '',
              seq.flicker_static != null ? (seq.flicker_static * 1000).toFixed(4) : '',
              seq.peak_flicker_idx != null ? seq.peak_flicker_idx + 1 : '',
            ]
              .map(esc)
              .join(',')
          );
        }
        const blob = new Blob(['\uFEFF' + lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        triggerBrowserDownload(url, `WildCat_Experiments_${Date.now()}.csv`);
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        notify(t('msg.labCsvExportedToast'), 'success');
      }

      watch(
        [() => labSelectedTaskIds.value.join(','), labEvalAssetId, labCkptMode],
        () => {
          if (topNav.value === 'lab' && labMode.value === 'analyze') {
            fetchLabCompare();
          }
        }
      );

      watch([labChartTab, () => labChartCfg.logScale, () => labChartCfg.smooth], () => {
        if (topNav.value === 'lab' && labMode.value === 'analyze') {
          nextTick(() => drawLabCompareChart());
        }
      });

      // ───────── Timers & Lifecycle ─────────
      watch(
        [topNav, () => assets.value.length],
        async ([nNav]) => {
          if (nNav === 'assets' && user.value) {
            for (const a of assets.value) {
              await ensureAssetFrames(a.id);
            }
          }
        }
      );

      watch([topNav, studioMode, labMode], ([nNav, nMode, nLabMode]) => {
        if (nNav === 'studio' && nMode === 'monitor') fetchTelemetry(true);
        if (nNav === 'help') loadHelpStaticData();
        if (nNav === 'admin' && adminToken.value) loadAdminData();
        if (nNav === 'lab' && nLabMode === 'analyze') {
          if (!labSelectedTaskIds.value.length && tasks.value.length) {
            selectFirstReadyLabGroup();
          } else {
            fetchLabCompare();
          }
        }
      });

      onMounted(() => {
        restoreSession();
        fetchGpuBenchmarks();
        if (typeof window !== 'undefined') {
          window.addEventListener('keydown', (ev) => {
            if (ev.key !== 'Escape') return;
            if (uploadModal.open) {
              if (!uploadModal.uploading) uploadModal.open = false;
              return;
            }
            if (roamModal.open) {
              if (!roamModal.loading) roamModal.open = false;
              return;
            }
            if (datasetPickerModal.open) {
              datasetPickerModal.open = false;
            }
          });
          let resizeRaf = 0;
          window.addEventListener('resize', () => {
            if (resizeRaf) return;
            resizeRaf = requestAnimationFrame(() => {
              resizeRaf = 0;
              if (topNav.value === 'studio' && studioMode.value === 'monitor') {
                drawLossChart();
                drawQualityChart();
              } else if (topNav.value === 'lab' && labMode.value === 'analyze') {
                drawLabCompareChart();
              }
            });
          });
        }

        // Poll queue, Studio telemetry & Lab evaluation progress every 1.8s
        let prevEvalBusy = false;
        setInterval(() => {
          if (!user.value) return;
          fetchQueueAndTasks();
          if (topNav.value === 'studio' && studioMode.value === 'monitor') {
            fetchTelemetry(false);
          }
          if (topNav.value === 'lab' && labMode.value === 'analyze' && labSelectedTaskIds.value.length) {
            const anyRunning = labCompare.runs.some((r) => r.status === 'running' || r.status === 'queued');
            if (labCompare.eval_runner.busy || prevEvalBusy || anyRunning) {
              prevEvalBusy = !!labCompare.eval_runner.busy;
              fetchLabCompare(true);
            }
          }
        }, 1800);

        // Sequence, 3x3 Contact Sheet & Lab Multi-Model player loop
        let lastViewerTick = 0;
        let lastContactTick = 0;
        let lastLabTick = 0;
        setInterval(() => {
          const now = performance.now();
          if (topNav.value === 'studio' && studioMode.value === 'monitor') {
            if (viewer.playing && viewer.frames.length > 1) {
              const intervalMs = 1000 / Math.min(60, Math.max(1, Number(viewer.fps) || 12));
              if (now - lastViewerTick >= intervalMs) {
                viewer.index = (viewer.index + 1) % viewer.frames.length;
                lastViewerTick = now;
              }
            }
            if (contactViewer.playing && contactViewer.frames.length > 1) {
              const intervalMs = 1000 / Math.min(60, Math.max(1, Number(contactViewer.fps) || 8));
              if (now - lastContactTick >= intervalMs) {
                contactViewer.index = (contactViewer.index + 1) % contactViewer.frames.length;
                lastContactTick = now;
              }
            }
          }
          if (topNav.value === 'lab' && labMode.value === 'analyze') {
            const tot = labSeqFrameCount.value;
            if (labViewer.playing && tot > 1) {
              const intervalMs = 1000 / Math.min(60, Math.max(1, Number(labViewer.fps) || 12));
              if (now - lastLabTick >= intervalMs) {
                labViewer.index = (labViewer.index + 1) % tot;
                lastLabTick = now;
                if (isLabPerFrameChart.value) drawLabCompareChart();
              }
            }
          }
        }, 30);

        if (typeof window !== 'undefined') {
          window.addEventListener('resize', () => {
            if (topNav.value === 'studio' && studioMode.value === 'monitor') {
              drawLossChart();
              drawQualityChart();
            }
            if (topNav.value === 'lab' && labMode.value === 'analyze') drawLabCompareChart();
          });
          window.addEventListener('popstate', () => {
            if (window.location && window.location.pathname === '/admin') {
              topNav.value = 'admin';
              if (adminToken.value) loadAdminData();
            } else if (topNav.value === 'admin') {
              topNav.value = 'studio';
            }
          });
        }
      });

      return {
        t,
        tHtml,
        availableLocales,
        currentLang,
        setLanguage,
        toasts,
        user,
        buildLabel,
        apiDocsOpen,
        machineName,
        roamModal,
        openRoamModal,
        copyMyRecoveryCode,
        downloadMyRecoveryTxt,
        submitRecoverWorkspace,
        restrictionsEnabled,
        isPrivilegedUser,
        canUseLab,
        canUseApi,
        canUseConverter,
        userMaxEpochs,
        userTaskQuotaReached,
        adminToken,
        adminPwdInput,
        adminAuthLoading,
        adminAuthError,
        pwdForm,
        doAdminLogin,
        logoutAdmin,
        changeAdminPassword,
        topNav,
        toolsSections,
        toolsSection,
        toolsTabLocked,
        workspaceReady,
        studioMode,
        labMode,
        initCategory,
        accelAll,
        accelAny,
        queue,
        tasks,
        assets,
        pairedAssets,
        evalSequenceAssets,
        selectedDatasetId,
        selectedDataset,
        checkpoints,
        selectedTaskId,
        extendTargetEpochs,
        extendInputFocused,
        extendEstimatedSteps,
        estimatedNewTaskDurationSec,
        activeEtaGpuProfile,
        gpuBenchDb,
        wizardQueuePreview,
        taskAheadCount,
        formatEta,
        helpTab,
        helpMainRef,
        switchHelpTab,
        helpPrevTab,
        helpNextTab,
        helpBench,
        openHelp,
        helpTabs,
        helpDocQuery,
        helpEvQuery,
        helpFocusId,
        helpReturn,
        helpGoBack,
        helpDocGroupsView,
        helpDocTotal,
        helpDocShown,
        helpEvGroupsView,
        helpEvTotal,
        helpEvShown,
        docTitle,
        docSummary,
        docHtml,
        hasDoc,
        evTitle,
        evSummary,
        evHtml,
        hasEv,
        openHelpDoc,
        openHelpEvidence,
        presetValueRows,
        applyHelpPreset,
        PRESET_IDS,
        benchTotalSpeedup,
        taskEpoch,
        form,
        channelPreset,
        effectiveBatchSize,
        isBatchClampedByFrames,
        stageCropPreview,
        advPanelOpen,
        activeAdvPreset,
        advItemOpen,
        advRawTexOpen,
        adv,
        advMathHtml,
        advLatexSrc,
        advCropGridOptions,
        isAdvModuleEffective,
        advActiveCount,
        advAnyItemExpanded,
        advStageStepsPreview,
        advHdrPreview,
        advExposurePreview,
        advEmaPreview,
        toggleAdvItem,
        toggleAdvAllItems,
        toggleRawLatex,
        copyLatexSource,
        resetAdvToNukeNative,
        applyAdvPreset,
        setCropGridAnchor,
        cropGridPhaseHint,
        onStageRatioInput,
        syncLossFamilyEnabled,
        onLossFamilyToggle,
        syncAugmentEnabled,
        onAugmentToggle,
        kfCommonBlockReason,
        kfCommonLimits,
        kfCommonRaftOffered,
        syncKfCommonParams,
        activeTaskKfCommon,
        kfCommonPct,
        kfCommonReasonText,
        kfCommonFlowWhy,
        isChannelPresetAllowed,
        channelOptionLabel,
        hasSelectedDatasetEvalSeq,
        selectedCkptMeta,
        isModelSizeLocked,
        isChannelsLocked,
        computedTotalSteps,
        initSummaryText,
        activeTask,
        telemetry,
        viewerModes,
        viewer,
        currentFrame,
        contactViewer,
        currentContactFrame,
        taskModels,
        latestValMse,
        chartCfg,
        lossCanvas,
        qualityView,
        qualityUi,
        qualityCanvas,
        qualityMeasures,
        qualityBar: QUALITY_BAR,
        qualityHoverPoint,
        qualityT,
        qualityStripCallout,
        onQualityChartMove,
        clearQualityHover,
        stepQualityHover,
        stepQualityFrame,
        logPreRef,
        logAutoScroll,
        uploadModal,
        datasetPickerModal,
        datasetPickerTitle,
        selectedLabEvalExplicitAsset,
        pickerBaseDatasets,
        pickerWithSeqCount,
        filteredPickerDatasets,
        openDatasetPicker,
        isPickerAssetSelected,
        pickDatasetItem,
        adminSettings,
        adminGpuList,
        adminEnabledGpuCount,
        adminUsers,
        adminTasks,
        newUser,
        openCreateWizard,
        selectTaskForMonitor,
        selectInitCategory,
        onCheckpointSelected,
        applyCkptPaths,
        onChannelPresetChange,
        onSelectDataset,
        onModelSizeChange,
        onBsModeChange,
        submitNewTrainingTask,
        resumeExistingTask,
        extendExistingTask,
        retryFailedTask,
        switchViewerMode,
        fetchTelemetry,
        drawLossChart,
        stepFrame,
        togglePlay,
        resetViewerZoom,
        onViewerWheel,
        onViewerMouseDown,
        onViewerMouseMove,
        onViewerMouseUp,
        stepContactFrame,
        toggleContactPlay,
        resetContactZoom,
        onContactWheel,
        onContactMouseDown,
        onContactMouseMove,
        onContactMouseUp,
        stopTask,
        deleteTask,
        downloadModel,
        downloadAllModels,
        copyTrainLog,
        openDatasetModal,
        openAttachSeqModal,
        detachEvalSequence,
        updateDatasetTaskType,
        assetFramesCache,
        getAssetScrubber,
        ensureAssetFrames,
        setAssetScrubMode,
        stepAssetFrame,
        currentAssetPairFrame,
        currentAssetSeqFrame,
        demoBadgeTitle,
        demoDescText,
        formatGpuOptionLabel,
        formatDatasetOptionLabel,
        formatLabDatasetOptionLabel,
        onUploadTaskTypeChange,
        onUploadStrategyChange,
        onPairFilesSelected,
        doUploadPairedDataset,
        importDemoAssets,
        useDatasetForTraining,
        useAssetForLabEval,
        deleteAsset,
        // Experiment Lab bindings
        UNGROUPED_LAB_KEY,
        activeLabDataset,
        hasLabEvalSequence,
        labStaged,
        labStagedGroups,
        labActiveStagedGroupId,
        activeStagedGroup,
        labTotalStagedCount,
        labNonEmptyStagedGroupsCount,
        labNote,
        labBatchName,
        labSubmitting,
        labActiveBatchId,
        labBatchGroups,
        labBatchTasks,
        labBatchStats,
        addNewLabStagedGroup,
        removeLabStagedGroup,
        moveLabStagedGroupOrder,
        moveLabStagedItemOrder,
        moveLabStagedItemToGroup,
        addCurrentToLabList,
        addNukeBaselineToLabList,
        labComboCfg,
        hasExplicitDatasetTaskType,
        detectedLabComboTaskType,
        effectiveLabComboTaskType,
        labComboSuitePreview,
        labComboKfNote,
        stageRecommendedComboSuite,
        loadStagedToForm,
        removeLabStaged,
        clearLabStaged,
        submitLabStagedGroup,
        submitAllLabStagedGroups,
        submitLabBatch,
        submitLabDirectOrStaged,
        openBatchInAnalyze,
        labNamedGroups,
        labGroupedTasks,
        labCollapsedGroups,
        labGroupCreateOpen,
        labNewGroupName,
        labCreateMoveSelected,
        labRenamingGroupId,
        labRenamingGroupName,
        isLabGroupCollapsed,
        toggleLabGroupCollapse,
        areAllLabGroupsCollapsed,
        toggleAllLabGroupsCollapse,
        isLabGroupAllSelected,
        selectOnlyLabGroup,
        toggleLabGroupSelection,
        openCreateLabGroup,
        confirmCreateLabGroup,
        startRenameLabGroup,
        cancelRenameLabGroup,
        confirmRenameLabGroup,
        moveLabGroupOrder,
        moveTaskWithinLabGroup,
        onSingleTaskMoveSelect,
        onBatchTaskMoveSelect,
        dissolveLabGroup,
        labSelectedTaskIds,
        labBaselineTaskId,
        labBaselineRun,
        labTaskFilter,
        labTaskSortMode,
        labArchiving,
        labDbSummary,
        labFilteredTasks,
        labEvalAssetId,
        labCkptMode,
        labCompare,
        labRunColor,
        toggleLabTaskSelection,
        selectAllCompletedForLab,
        clearLabSelection,
        archiveSelectedLabTaskFiles,
        fetchLabCompare,
        triggerLabEvaluation,
        cancelLabEvaluation,
        labChartTab,
        labChartCfg,
        labCompareCanvas,
        labChartHover,
        isLabPerFrameChart,
        drawLabCompareChart,
        onLabChartMouseMove,
        onLabChartMouseLeave,
        onLabChartClick,
        getRunMetric,
        isBestLabMetric,
        formatLabDelta,
        labViewer,
        labEvaluatedSeqRuns,
        labSeqHasPairedErr,
        labSeqFrameCount,
        labGlobalPeakIdx,
        labCurrentSourceFrame,
        getLabRunFrame,
        jumpLabToFrame,
        stepLabFrame,
        resetLabZoom,
        onLabWheel,
        onLabMouseDown,
        onLabMouseMove,
        onLabMouseUp,
        exportLabCsv,
        exportAnalyticsDb,
        copyMyApiKey,
        regenerateApiKey,
        switchToAdmin,
        goBackToStudio,
        loadAdminData,
        saveAdminSettings,
        toggleAdminGpuEnabled,
        enableAllAdminGpus,
        setAdminGpuRole,
        toggleTempUserRestrictions,
        saveDefaultTempQuotas,
        saveAdminStorageSettings,
        resetAdminStorageDefault,
        adminCreateUser,
        impersonateUserFromAdmin,
        adminSaveUserQuota,
        adminResetUserUsedTasks,
        adminToggleUserPerm,
        adminCopyUserRoam,
        adminRegenUserRoam,
        adminPromptEditNote,
        adminToggleUserActive,
        adminDeleteUser,
        statusBadgeText,
        statusLabel,
        progressPct,
      };
    },
  })
    .component('help-tip', HelpTip)
    .mount('#app');
}

mountWildCatApp();
