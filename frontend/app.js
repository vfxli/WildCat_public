import { createApp, ref, reactive, computed, onMounted, nextTick, watch } from '/static/vendor/vue.esm.js';
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
  el_ema: String.raw`\tilde{\theta}_k = \beta_k\,\tilde{\theta}_{k-1} + (1 - \beta_k)\,\theta_k, \qquad \beta_k = \min\!\left(\beta_{\mathrm{ema}},\; \frac{1 + k}{10 + k}\right)`,
  el_window: String.raw`N_{\mathrm{eff}} \approx \frac{1}{1 - \beta_{\mathrm{ema}}}, \qquad T_{1/2} = \frac{\ln 2}{-\ln \beta_{\mathrm{ema}}} \approx \frac{0.693}{1 - \beta_{\mathrm{ema}}}, \qquad \theta_k \longleftarrow \operatorname{Adam}\!\bigl(\eta_{\mathrm{Nuke}}(k)\bigr)`,
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

function createDefaultAdvConfig() {
  return {
    // 01. Crop origin sub-pixel phase grid (eliminates Nuke 4px bottom-anchor phase mismatch)
    crop_grid: {
      enabled: false,
      full_res_grid: 1, // 1 | 2 | 4 (Nuke native = 4)
    },
    // 02. Composite loss: Charbonnier + L1 Spatial Edge Gradient + ACEScg HDR Highlight Damping
    loss_family: {
      enabled: false,
      charbonnier_ratio: 0.25,
      charb_eps: 0.001,
      grad_weight: 0.05,
      hdr_alpha: 0.5,
    },
    // 03. Multi-resolution stage duration split & stage transition LR soft warmup
    stage_schedule: {
      enabled: false,
      r_quarter: 15,
      r_half: 25,
      r_full: 60,
      warmup_steps: 200,
    },
    // 04. Stage-reset adaptive EMA shadow weights (preserves 100% Nuke native poly(0.3) optimizer trajectory)
    ema_lr: {
      enabled: false,
      ema_decay: 0.9995,
    },
    // 05. Semantic-preserving optical augmentation (pure HFlip=0.5, EV=0.0 per blind-test ablation)
    augment: {
      enabled: false,
      hflip_prob: 0.5,
      exposure_ev: 0.0,
    },
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
      const topNav = ref(initialNav); // 'studio' | 'lab' | 'assets' | 'api' | 'help' | 'admin'
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
      const HELP_TAB_ORDER = ['quickstart', 'verdicts', 'presets', 'report', 'database', 'benchmarks'];
      const helpTab = ref('quickstart');
      const helpMainRef = ref(null);
      const helpCurveTab = ref('focus'); // 'focus' | 'degrain' | 'scar'
      const helpCompareShotFilter = ref('all');
      const helpDeepCurveGroup = ref('focus_6way');
      const helpDeepShowSeen = ref(false);
      const helpLedgerShot = ref('all');
      const helpLedgerModel = ref('all');
      const helpLedgerSort = ref('default');
      const helpLedgerQuery = ref('');
      const DEFAULT_HELP_PINS = ['Exp_scar_L_nuke_1M', 'Exp_rf_scar_L_crop384_1M', 'Exp_sub_scar_aug_flip_500k'];
      const helpPinnedRunIds = ref([...DEFAULT_HELP_PINS]);
      const helpBenchRaw = ref(null);
      const helpBenchError = ref(false);
      const helpGuideRaw = ref(null);
      const helpGuideLoading = ref(false);
      const gpuBenchDb = ref(null);

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

      const helpGuide = computed(() => helpGuideRaw.value);

      const helpTelVars = computed(() => {
        const tel = (helpGuideRaw.value && helpGuideRaw.value.telemetry) || {};
        return {
          epochsYi: tel.raw_epochs_yi != null ? tel.raw_epochs_yi : '1.52',
          shadowYi: tel.shadow_epochs_yi != null ? tel.shadow_epochs_yi : '7.42',
          epochsM: tel.raw_epochs_m != null ? tel.raw_epochs_m : '151.6',
          shadowM: tel.shadow_epochs_m != null ? tel.shadow_epochs_m : '741.5',
          h16: tel.rtx5090_16_hours != null ? tel.rtx5090_16_hours : '23.9',
          h1: tel.rtx5090_1_hours != null ? tel.rtx5090_1_hours : '382.9',
        };
      });

      function helpLoc(item) {
        if (!item || typeof item !== 'object') return '';
        const isEn = String(currentLang.value || '').toLowerCase().startsWith('en');
        return isEn ? (item.en || item.zh || '') : (item.zh || item.en || '');
      }

      const filteredShotDeepReports = computed(() => {
        const g = helpGuideRaw.value;
        const list = (g && g.shot_deep_reports) || [];
        if (helpCompareShotFilter.value === 'all') return list;
        return list.filter((s) => s.dataset === helpCompareShotFilter.value);
      });

      const helpDeepCurveChart = computed(() => {
        const g = helpGuideRaw.value;
        const groups = (g && g.multi_curve_groups) || [];
        if (!groups.length) return null;
        const active = groups.find((x) => x.id === helpDeepCurveGroup.value) || groups[0];
        const X0 = 56;
        const W = 710;
        const Y0 = 22;
        const H = 220;
        const yMin = Number(active.y_min || 10);
        const yMax = Number(active.y_max || 65);
        const maxK = Number(active.max_k || 1000);
        const mapX = (stepK) => Number((X0 + (Math.max(0, Math.min(maxK, Number(stepK || 0))) / maxK) * W).toFixed(1));
        const mapY = (val) => {
          const clamped = Math.max(yMin, Math.min(yMax, Number(val || yMin)));
          return Number((Y0 + H - ((clamped - yMin) / Math.max(0.1, yMax - yMin)) * H).toFixed(1));
        };
        const series = [];
        for (const s of active.series || []) {
          const dots = (s.pts || []).map((p) => ({
            step_k: p.step_k,
            val: Number(p.unseen || 0),
            seen: Number(p.seen || 0),
            x: mapX(p.step_k),
            y: mapY(p.unseen),
          }));
          series.push({
            id: s.run_id,
            label: helpLoc(s.label),
            color: s.color,
            dashed: Boolean(s.dashed),
            opacity: 1.0,
            strokeWidth: s.dashed ? 2.0 : 2.6,
            pointsStr: dots.map((d) => `${d.x},${d.y}`).join(' '),
            dots,
          });
          if (helpDeepShowSeen.value) {
            const seenDots = (s.pts || []).map((p) => ({
              step_k: p.step_k,
              val: Number(p.seen || 0),
              x: mapX(p.step_k),
              y: mapY(p.seen),
            }));
            series.push({
              id: `${s.run_id}_seen`,
              label: `${helpLoc(s.label)} [Seen]`,
              color: s.color,
              dashed: true,
              opacity: 0.45,
              strokeWidth: 1.5,
              pointsStr: seenDots.map((d) => `${d.x},${d.y}`).join(' '),
              dots: [],
            });
          }
        }
        const gridLines = (active.y_ticks || []).map((tv) => ({
          val: tv,
          y: mapY(tv),
        }));
        const xTicks = [0, 200, 333, 500, 667, 800, 1000].filter((k) => k <= maxK).map((k) => ({
          step_k: k,
          x: mapX(k),
        }));
        return {
          group: active,
          x0: X0,
          w: W,
          y0: Y0,
          h: H,
          xStage1: mapX(maxK * 0.3333),
          xStage2: mapX(maxK * 0.6667),
          gridLines,
          xTicks,
          series,
        };
      });

      function isHelpRunPinned(runId) {
        return helpPinnedRunIds.value.includes(runId);
      }

      function toggleHelpPinRun(runId) {
        const cur = [...helpPinnedRunIds.value];
        const idx = cur.indexOf(runId);
        if (idx >= 0) {
          cur.splice(idx, 1);
        } else {
          if (cur.length >= 6) cur.pop();
          cur.push(runId);
        }
        helpPinnedRunIds.value = cur;
      }

      function setHelpPinBase(runId) {
        const cur = helpPinnedRunIds.value.filter((x) => x !== runId);
        helpPinnedRunIds.value = [runId, ...cur];
      }

      function clearHelpPins() {
        helpPinnedRunIds.value = [];
      }

      function resetHelpDemoPins() {
        helpPinnedRunIds.value = [...DEFAULT_HELP_PINS];
      }

      const filteredHelpLedger = computed(() => {
        const g = helpGuideRaw.value;
        const all = (g && g.all_runs_ledger) || [];
        const q = String(helpLedgerQuery.value || '').trim().toLowerCase();
        const isEn = String(currentLang.value || '').toLowerCase().startsWith('en');
        const filtered = all.filter((r) => {
          if (helpLedgerShot.value !== 'all' && r.dataset !== helpLedgerShot.value) return false;
          if (helpLedgerModel.value !== 'all' && r.model !== helpLedgerModel.value) return false;
          if (q) {
            const hay = `${r.run_id} ${r.dataset} ${isEn ? r.shot_en : r.shot_zh} ${isEn ? r.config_en : r.config_zh} ${r.model}`.toLowerCase();
            if (!hay.includes(q)) return false;
          }
          return true;
        });
        const mode = helpLedgerSort.value;
        if (mode === 'default') return filtered;
        const copy = [...filtered];
        const numDesc = (a, b, k) => {
          const av = a[k] == null ? -1e9 : Number(a[k]);
          const bv = b[k] == null ? -1e9 : Number(b[k]);
          return bv - av;
        };
        const numAsc = (a, b, k) => {
          const av = a[k] == null ? 1e9 : Number(a[k]);
          const bv = b[k] == null ? 1e9 : Number(b[k]);
          return av - bv;
        };
        if (mode === 'unseen_desc') copy.sort((a, b) => numDesc(a, b, 'unseen_aces'));
        else if (mode === 'peak_desc') copy.sort((a, b) => numDesc(a, b, 'peak_unseen'));
        else if (mode === 'roi_desc') copy.sort((a, b) => numDesc(a, b, 'unseen_roi'));
        else if (mode === 'hfi_asc') copy.sort((a, b) => numAsc(a, b, 'seq_hfi'));
        else if (mode === 'time_asc') copy.sort((a, b) => numAsc(a, b, 'minutes'));
        return copy;
      });

      const pinnedComparisonRows = computed(() => {
        const g = helpGuideRaw.value;
        const all = (g && g.all_runs_ledger) || [];
        const byId = new Map(all.map((r) => [r.run_id, r]));
        const items = helpPinnedRunIds.value.map((id) => byId.get(id)).filter(Boolean);
        if (!items.length) return [];
        const base = items[0];
        return items.map((r, idx) => {
          const isBase = idx === 0;
          const dUns = !isBase && r.unseen_aces != null && base.unseen_aces != null
            ? Number((r.unseen_aces - base.unseen_aces).toFixed(2))
            : null;
          const dPeak = !isBase && r.peak_unseen != null && base.peak_unseen != null
            ? Number((r.peak_unseen - base.peak_unseen).toFixed(2))
            : null;
          const dRoi = !isBase && r.unseen_roi != null && base.unseen_roi != null
            ? Number((r.unseen_roi - base.unseen_roi).toFixed(2))
            : null;
          const dHfiPct = !isBase && r.seq_hfi != null && base.seq_hfi != null && base.seq_hfi > 0
            ? Number((((r.seq_hfi - base.seq_hfi) / base.seq_hfi) * 100).toFixed(1))
            : null;
          const dTimePct = !isBase && r.minutes != null && base.minutes != null && base.minutes > 0
            ? Number((((r.minutes - base.minutes) / base.minutes) * 100).toFixed(1))
            : null;
          return {
            ...r,
            is_pin_base: isBase,
            delta_unseen: dUns,
            delta_peak: dPeak,
            delta_roi: dRoi,
            delta_hfi_pct: dHfiPct,
            delta_time_pct: dTimePct,
          };
        });
      });

      function helpHeroBarPct(dbVal) {
        const v = Number(dbVal || 0);
        return Math.max(16, Math.min(100, ((v - 24) / 42) * 100)).toFixed(1);
      }

      function helpTriadBarPct(dbVal) {
        const v = Number(dbVal || 0);
        return Math.max(10, Math.min(100, ((v - 8) / 72) * 100)).toFixed(1);
      }

      const helpTrajectoryChart = computed(() => {
        const g = helpGuideRaw.value;
        const tr = (g && g.trajectories) || {};
        const isEn = String(currentLang.value || '').toLowerCase().startsWith('en');
        const X0 = 54;
        const W = 680;
        const Y0 = 24;
        const H = 180;

        let tab = helpCurveTab.value || 'focus';
        let yMin = 22;
        let yMax = 46;
        let yTicks = [24, 30, 36, 42, 46];
        let rawSeries = [];
        let noteKey = 'help.visCurveFocusNote';

        if (tab === 'degrain') {
          yMin = 38.0;
          yMax = 42.4;
          yTicks = [38, 39, 40, 41, 42];
          noteKey = 'help.visCurveDegrainNote';
          rawSeries = [
            {
              id: 'wc_deg',
              label: isEn ? 'WildCat Small Combo (Unseen)' : 'WildCat 小模型强化 (盲测真本事)',
              pts: tr.degrain_small_combo || [],
              field: 'unseen',
              color: '#34d399',
              dashed: false,
              pinStepK: 700,
              pinText: isEn ? '★ 700k Peak 41.78 dB (Full-Frame Best)' : '★ 70万步峰值 41.78 dB (整图直出最优)',
            },
            {
              id: 'nuke_deg',
              label: isEn ? 'Nuke 17 Default (Unseen)' : 'Nuke 17 默认 (盲测真本事)',
              pts: tr.degrain_small_nuke || [],
              field: 'unseen',
              color: '#94a3b8',
              dashed: true,
              pinStepK: null,
              pinText: '',
            },
          ];
        } else if (tab === 'scar') {
          yMin = 26.8;
          yMax = 32.0;
          yTicks = [27, 28, 29, 30, 31, 32];
          noteKey = 'help.visCurveScarNote';
          rawSeries = [
            {
              id: 'wc_scar',
              label: isEn ? 'Crop 384 + Robust Loss (Unseen)' : '黄金 Crop 384 + 抗噪损失 (盲测真本事)',
              pts: tr.scar_crop384_combo || [],
              field: 'unseen',
              color: '#34d399',
              dashed: false,
              pinStepK: 600,
              pinText: isEn ? '★ 600k Peak 31.31 dB (Keymix Best)' : '★ 60万步半图末峰值 31.31 dB (Keymix最优)',
            },
            {
              id: 'bad_scar',
              label: isEn ? '❌ Naked 1px + Pure MSE (Crashes in 1/1)' : '❌ 单开 1px 不配抗噪损失 (进1/1背底噪暴跌)',
              pts: tr.scar_grid1_only_bad || [],
              field: 'unseen',
              color: '#fb7185',
              dashed: true,
              pinStepK: null,
              pinText: '',
            },
          ];
        } else {
          yMin = 22.5;
          yMax = 46.5;
          yTicks = [24, 30, 36, 42, 46];
          noteKey = 'help.visCurveFocusNote';
          rawSeries = [
            {
              id: 'wc_focus_uns',
              label: isEn ? 'WildCat Large Combo (Blind Unseen)' : 'WildCat 大模型组合 (盲测真本事 Unseen)',
              pts: tr.focus_large_combo || [],
              field: 'unseen',
              color: '#34d399',
              dashed: false,
              pinStepK: 1000,
              pinText: isEn ? '★ 1000k Peak 43.55 dB (+9.72 dB)' : '★ 100万步 43.55 dB (净涨 +9.72 dB)',
            },
            {
              id: 'wc_focus_seen',
              label: isEn ? 'Training Seen (Moves in lockstep — no overfit)' : '训练帧 Seen (同步上涨 · 零死记硬背)',
              pts: tr.focus_large_combo || [],
              field: 'seen',
              color: '#818cf8',
              dashed: true,
              pinStepK: null,
              pinText: '',
            },
          ];
        }

        const mapX = (pct) => Number((X0 + (Math.max(0, Math.min(100, pct)) / 100) * W).toFixed(1));
        const mapY = (val) => {
          const clamped = Math.max(yMin, Math.min(yMax, Number(val || yMin)));
          return Number((Y0 + H - ((clamped - yMin) / Math.max(0.1, yMax - yMin)) * H).toFixed(1));
        };

        const series = rawSeries.map((s) => {
          const dots = (s.pts || []).map((p) => {
            const val = Number(p[s.field] || 0);
            return {
              step_k: p.step_k,
              pct: p.pct,
              val,
              x: mapX(p.pct),
              y: mapY(val),
            };
          });
          const pointsStr = dots.map((d) => `${d.x},${d.y}`).join(' ');
          const pinDot = s.pinStepK ? dots.find((d) => d.step_k === s.pinStepK) || dots[dots.length - 1] : null;
          return {
            id: s.id,
            label: s.label,
            color: s.color,
            dashed: s.dashed,
            pointsStr,
            dots,
            pinDot,
            pinText: s.pinText,
          };
        });

        const gridLines = yTicks.map((tv) => ({
          val: tv,
          y: mapY(tv),
        }));

        return {
          x0: X0,
          w: W,
          y0: Y0,
          h: H,
          xStage1: Number((X0 + W * 0.3333).toFixed(1)),
          xStage2: Number((X0 + W * 0.6667).toFixed(1)),
          gridLines,
          series,
          noteText: t(noteKey),
        };
      });

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
          notes: [
            t('help.benchNote1'),
            t('help.benchNote2'),
            t('help.benchNote3'),
            t('help.benchNote4'),
          ],
        };
      });

      async function loadHelpStaticData() {
        if (helpGuideLoading.value) return;
        helpGuideLoading.value = true;
        const tasksArr = [];
        if (!helpGuideRaw.value) {
          tasksArr.push(
            fetch('/static/empirical_guide.json?v=20261114_audit1')
              .then((r) => (r.ok ? r.json() : null))
              .then((d) => {
                if (d) helpGuideRaw.value = d;
              })
              .catch(() => {})
          );
        }
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
        helpGuideLoading.value = false;
      }

      async function openHelp(tabId) {
        if (typeof tabId === 'string' && HELP_TAB_ORDER.includes(tabId)) {
          helpTab.value = tabId;
        }
        topNav.value = 'help';
        await loadHelpStaticData();
        if (helpMainRef.value) helpMainRef.value.scrollTop = 0;
      }

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
        auto_extend_optimal: true,
        stop_after_optimal_epochs: 0,
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
      });
      const advRawTexOpen = reactive({});
      const adv = reactive(createDefaultAdvConfig());
      const advMathHtml = ADV_MATH_HTML;
      const advLatexSrc = ADV_LATEX;

      const ADV_KEYS = ['crop_grid', 'loss_family', 'stage_schedule', 'ema_lr', 'augment'];

      const advCropGridOptions = computed(() => [
        { v: 1, title: t('adv.cg.opt1Title'), desc: t('adv.cg.opt1Desc') },
        { v: 2, title: t('adv.cg.opt2Title'), desc: t('adv.cg.opt2Desc') },
        { v: 4, title: t('adv.cg.opt4Title'), desc: t('adv.cg.opt4Desc') },
      ]);

      const advEmaPresets = computed(() => [
        { d: 0.999, lbl: t('adv.el.presetShort') },
        { d: 0.9995, lbl: t('adv.el.presetStd') },
        { d: 0.9998, lbl: t('adv.el.presetLong') },
      ]);

      function isAdvModuleEffective(k) {
        if (!adv[k] || !adv[k].enabled) return false;
        if (k === 'crop_grid') {
          return Number(adv.crop_grid.full_res_grid || 4) !== 4;
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
        return true;
      }

      const advActiveCount = computed(() => {
        return ADV_KEYS.reduce((n, k) => n + (isAdvModuleEffective(k) ? 1 : 0), 0);
      });

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
            if (oc === 4) {
              adv.loss_family.charbonnier_ratio = 0.3;
              adv.loss_family.grad_weight = 0.15;
              adv.loss_family.hdr_alpha = 0.35;
            } else {
              adv.loss_family.charbonnier_ratio = 0.25;
              adv.loss_family.grad_weight = 0.05;
              adv.loss_family.hdr_alpha = oc === 1 ? 0 : 0.5;
            }
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
      }

      function toggleAdvAllItems(openState) {
        const target = typeof openState === 'boolean' ? openState : !advAnyItemExpanded.value;
        for (const k of ADV_KEYS) {
          advItemOpen[k] = target;
        }
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
          form.stop_after_optimal_epochs = 0;
          notify(t('msg.resetAdvDone'), 'info');
        }
      }

      function applyEmpiricalPresetCore(presetId) {
        form.stop_after_optimal_epochs = 0;
        const clampEp = (ep) => (userMaxEpochs.value > 0 ? Math.min(ep, userMaxEpochs.value) : ep);
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

        if (presetId === 'cleanup_grain') {
          setPresetModelAndCh('Small', '3->3');
          form.epochs = clampEp(300000);
          form.crop_size = safeCrop(256);
          adv.crop_grid.enabled = true;
          adv.crop_grid.full_res_grid = 1;
          adv.loss_family.enabled = true;
          adv.loss_family.charbonnier_ratio = 0.25;
          adv.loss_family.grad_weight = 0.05;
          adv.loss_family.hdr_alpha = Number(form.out_channels) === 1 ? 0 : 0.5;
          adv.stage_schedule.enabled = false;
          adv.ema_lr.enabled = true;
          adv.ema_lr.ema_decay = 0.9995;
          adv.augment.enabled = true;
          adv.augment.hflip_prob = 0.5;
          adv.augment.exposure_ev = 0.0;
        } else if (presetId === 'keying_despill') {
          setPresetModelAndCh('Small', '3->4');
          form.epochs = clampEp(500000);
          form.crop_size = safeCrop(256);
          adv.crop_grid.enabled = true;
          adv.crop_grid.full_res_grid = 1;
          adv.loss_family.enabled = true;
          adv.loss_family.charbonnier_ratio = 0.3;
          adv.loss_family.grad_weight = 0.15;
          adv.loss_family.hdr_alpha = Number(form.out_channels) === 1 ? 0 : 0.35;
          adv.stage_schedule.enabled = false;
          adv.ema_lr.enabled = true;
          adv.ema_lr.ema_decay = 0.9995;
          adv.augment.enabled = false;
        } else if (presetId === 'bokeh_focus') {
          setPresetModelAndCh('Large', '3->3');
          form.epochs = clampEp(500000);
          form.crop_size = safeCrop(256);
          adv.crop_grid.enabled = true;
          adv.crop_grid.full_res_grid = 1;
          adv.loss_family.enabled = true;
          adv.loss_family.charbonnier_ratio = 0.25;
          adv.loss_family.grad_weight = 0.1;
          adv.loss_family.hdr_alpha = Number(form.out_channels) === 1 ? 0 : 0.5;
          adv.stage_schedule.enabled = true;
          adv.stage_schedule.r_quarter = 15;
          adv.stage_schedule.r_half = 25;
          adv.stage_schedule.r_full = 60;
          adv.stage_schedule.warmup_steps = 200;
          adv.ema_lr.enabled = true;
          adv.ema_lr.ema_decay = 0.9995;
          adv.augment.enabled = false;
        } else if (presetId === 'degrain_clean') {
          setPresetModelAndCh('Small', '3->3');
          form.epochs = clampEp(400000);
          form.crop_size = safeCrop(256);
          adv.crop_grid.enabled = true;
          adv.crop_grid.full_res_grid = 1;
          adv.loss_family.enabled = true;
          adv.loss_family.charbonnier_ratio = 0.25;
          adv.loss_family.grad_weight = 0.05;
          adv.loss_family.hdr_alpha = Number(form.out_channels) === 1 ? 0 : 0.5;
          adv.stage_schedule.enabled = false;
          adv.ema_lr.enabled = true;
          adv.ema_lr.ema_decay = 0.9995;
          adv.augment.enabled = false;
        } else if (presetId === 'roto_matte') {
          setPresetModelAndCh('Small', '3->1');
          form.epochs = clampEp(600000);
          form.crop_size = safeCrop(256);
          adv.crop_grid.enabled = true;
          adv.crop_grid.full_res_grid = 1;
          adv.loss_family.enabled = true;
          adv.loss_family.charbonnier_ratio = 0.25;
          adv.loss_family.grad_weight = 0.05;
          adv.loss_family.hdr_alpha = 0.0;
          adv.stage_schedule.enabled = false;
          adv.ema_lr.enabled = true;
          adv.ema_lr.ema_decay = 0.9995;
          adv.augment.enabled = false;
        } else {
          return false;
        }
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
        notify(t('msg.presetEmpiricalLoaded', { name: t(`adv.preset_${presetId}`) }), 'success');
      }

      function applyHelpRecipe(rec) {
        if (!rec || !rec.id) return;
        initCategory.value = 'scratch';
        form.initial_weights_mode = 'None';
        form.checkpoint_path = '';
        applyEmpiricalPresetCore(rec.id);
        advPanelOpen.value = true;
        studioMode.value = 'create';
        topNav.value = 'studio';
        notify(t('msg.helpRecipeLoadedToast', { name: helpLoc(rec.title) }), 'success');
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

      const advEmaPreview = computed(() => {
        const d = Math.max(0.9, Math.min(0.99999, Number(adv.ema_lr.ema_decay || 0.9995)));
        const effSteps = Math.round(1 / Math.max(1e-6, 1 - d));
        const halfLife = Math.round(Math.LN2 / Math.max(1e-6, -Math.log(d)));
        return { decay: d, effSteps, halfLife };
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
        recommendations: null,
        patience_stop_info: null,
        evaluatingRec: false,
      });
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
        form.stop_after_optimal_epochs = 0;
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
        form.stop_after_optimal_epochs = 0;
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
        }
        taskModels.recommendations = null;
        taskModels.patience_stop_info = null;
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
          const emult = av.ema_lr && av.ema_lr.enabled ? Number(c.adv_ema_mult || 1.015) : 1.0;
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
            ema_lr: { enabled: isAdvModuleEffective('ema_lr') },
            augment: { enabled: isAdvModuleEffective('augment') },
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
            auto_extend_optimal: !!(hasSelectedDatasetEvalSeq.value && form.auto_extend_optimal),
            stop_after_optimal_epochs:
              (topNav.value === 'studio' || !hasSelectedDatasetEvalSeq.value) && !resume
                ? 0
                : Math.max(0, Math.floor(Number(form.stop_after_optimal_epochs || 0))),
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
        form.auto_extend_optimal = typeof tr.auto_extend_optimal === 'boolean' ? tr.auto_extend_optimal : true;
        form.stop_after_optimal_epochs = 0;
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
          taskModels.recommendations = md.recommendations || null;
          taskModels.patience_stop_info = md.patience_stop_info || null;

          await nextTick();
          drawLossChart();
          if (logAutoScroll.value && logPreRef.value && (resetToLatest || tailChanged)) {
            logPreRef.value.scrollTop = logPreRef.value.scrollHeight;
          }
        } catch (_) {}
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
          ctx.fillText(`Step ${stVal}`, gx, H - 8);
        }

        // Multi-Resolution Stage Boundary Markers (1/4 -> 1/2 -> 1/1)
        const tCfg = (activeTask.value && activeTask.value.config && activeTask.value.config.train) || {};
        const stAdv =
          (activeTask.value &&
            activeTask.value.config &&
            activeTask.value.config.advanced &&
            activeTask.value.config.advanced.stage_schedule) ||
          null;
        const totSt = Number((activeTask.value && activeTask.value.total_steps) || tCfg.total_steps || 0);
        if (tCfg.multi_resolution !== false && totSt >= 9) {
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
          ctx.save();
          ctx.setLineDash([4, 4]);
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.45)';
          ctx.fillStyle = 'rgba(251, 191, 36, 0.85)';
          ctx.font = '10px monospace';
          ctx.textAlign = 'left';
          [
            { step: s1, label: '1/4 → 1/2' },
            { step: s2, label: '1/2 → 1/1' },
          ].forEach((b) => {
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

      async function evaluateTaskRecommendations(taskId, force = false) {
        if (!taskId || taskModels.evaluatingRec) return;
        taskModels.evaluatingRec = true;
        try {
          notify(t('msg.evaluatingRecommendationsToast'), 'info');
          const q = force ? '?force=true' : '';
          const res = await apiFetch(`/api/v1/tasks/${taskId}/recommend-models${q}`, {
            method: 'POST',
          });
          const md = res.models || {};
          taskModels.nuke_prefix = md.nuke_prefix || taskModels.nuke_prefix;
          taskModels.zip_filename = md.zip_filename || taskModels.zip_filename;
          taskModels.count = Number(md.count || 0);
          taskModels.items = md.items || [];
          taskModels.has_eval_sequence = !!md.has_eval_sequence;
          taskModels.recommendations = md.recommendations || null;
          taskModels.patience_stop_info = md.patience_stop_info || null;
          notify(t('msg.recommendationsReadyToast'), 'success');
        } catch (e) {
          notify(e.message, 'error');
        } finally {
          taskModels.evaluatingRec = false;
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

      function openDatasetPicker(mode = 'train') {
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
        if (av.crop_grid && av.crop_grid.enabled && Number(av.crop_grid.full_res_grid || 4) !== 4) {
          nonDefault = true;
          tags.push(`Grid=${av.crop_grid.full_res_grid}px`);
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
        const stopOptEp = Math.max(0, Math.floor(Number(tr.stop_after_optimal_epochs || 0)));
        if (stopOptEp > 0) {
          tags.push(`OptStop(+${stopOptEp}ep)`);
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
        baseCfg.train.stop_after_optimal_epochs = 0;
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
        formSnap.stop_after_optimal_epochs = 0;
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
      const COMBO_DEFAULT_STOP_EPOCHS = {
        cleanup_grain: 60000,
        roto_matte: 100000,
        degrain_clean: 80000,
        keying_despill: 100000,
        bokeh_focus: 100000,
      };

      const VALID_COMBO_TASK_TYPES = [
        'cleanup_grain',
        'keying_despill',
        'bokeh_focus',
        'degrain_clean',
        'roto_matte',
      ];

      const labComboCfg = reactive({
        panelOpen: false,
        previewOpen: true,
        taskType: 'auto', // 'auto' | 'cleanup_grain' | 'keying_despill' | 'bokeh_focus' | 'degrain_clean' | 'roto_matte'
        scope: 'standard', // 'compact' (4) | 'standard' (6) | 'thorough' (8)
        stopAfterOptimalOn: true,
        stopAfterOptimalEpochs: 60000,
        stopEpochsUserTouched: false,
        autoExtendOn: true,
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
        if (Number(ds.avg_mae || 0) > 0 && Number(ds.avg_mae) < 0.012) return 'degrain_clean';
        return 'cleanup_grain';
      });

      const effectiveLabComboTaskType = computed(() => {
        return labComboCfg.taskType === 'auto' ? detectedLabComboTaskType.value : labComboCfg.taskType;
      });

      watch(
        effectiveLabComboTaskType,
        (nextType) => {
          if (!labComboCfg.stopEpochsUserTouched) {
            labComboCfg.stopAfterOptimalEpochs = COMBO_DEFAULT_STOP_EPOCHS[nextType] || 50000;
          }
        },
        { immediate: true }
      );

      function setLabComboPatiencePreset(ep) {
        labComboCfg.stopAfterOptimalOn = true;
        labComboCfg.stopAfterOptimalEpochs = Math.max(100, Math.floor(Number(ep) || 50000));
        labComboCfg.stopEpochsUserTouched = true;
      }

      function onLabComboPatienceInput() {
        labComboCfg.stopEpochsUserTouched = true;
      }

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

        let specs = [];
        if (effType === 'cleanup_grain') {
          specs = [
            {
              code: 'Anchor_FlipClean',
              role: 'anchor',
              reasonKey: 'lab.comboReason_cg_1',
              model_size: pickModel('Small'),
              epochs: clampEp(300000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'Med_PeakPSNR',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_cg_2',
              model_size: pickModel('Medium'),
              epochs: clampEp(220000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'Med_WideCtx',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_cg_3',
              model_size: pickModel('Medium'),
              epochs: clampEp(250000),
              crop_size: cropWide,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'CleanPlate_NoFlip',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_cg_4',
              model_size: pickModel('Small'),
              epochs: clampEp(400000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'CrispEdge_G12',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_cg_5',
              model_size: pickModel('Small'),
              epochs: clampEp(300000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.12, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'Large_WideCtx',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_cg_6',
              model_size: pickModel('Large'),
              epochs: clampEp(250000),
              crop_size: cropWide,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'PureGrid1_MSE',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_cg_7',
              model_size: pickModel('Small'),
              epochs: clampEp(350000),
              crop_size: cropTight,
              grid: 1,
              loss: null,
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'NukeBase',
              role: 'control',
              reasonKey: 'lab.comboReason_nuke',
              nukeNative: true,
              model_size: pickModel('Small'),
              epochs: clampEp(300000),
              crop_size: clampCrop(256),
            },
          ];
        } else if (effType === 'keying_despill') {
          specs = [
            {
              code: 'Anchor_Key',
              role: 'anchor',
              reasonKey: 'lab.comboReason_kd_1',
              model_size: pickModel('Small'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.15, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'Balanced_C25G05',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_kd_2',
              model_size: pickModel('Small'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'DeepHair_750k',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_kd_3',
              model_size: pickModel('Small'),
              epochs: clampEp(750000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.15, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'CapMed_Key',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_kd_4',
              model_size: pickModel('Medium'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.1, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'LinearAlpha_H0',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_kd_5',
              model_size: pickModel('Small'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.15, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'WideSpill_Ctx',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_kd_6',
              model_size: pickModel('Small'),
              epochs: clampEp(500000),
              crop_size: cropWide,
              grid: 1,
              loss: { cr: 0.25, gw: 0.12, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'SymFlip_Key',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_kd_7',
              model_size: pickModel('Small'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.15, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'NukeBase',
              role: 'control',
              reasonKey: 'lab.comboReason_nuke',
              nukeNative: true,
              model_size: pickModel('Small'),
              epochs: clampEp(500000),
              crop_size: clampCrop(256),
            },
          ];
        } else if (effType === 'bokeh_focus') {
          specs = [
            {
              code: 'Anchor_Bokeh',
              role: 'anchor',
              reasonKey: 'lab.comboReason_bf_1',
              model_size: pickModel('Large'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.1, ha: cOut === 1 ? 0 : 0.5 },
              stage: [15, 25, 60],
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'EqualStage_Large',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_bf_2',
              model_size: pickModel('Large'),
              epochs: clampEp(600000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'WideDisc_Ctx',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_bf_3',
              model_size: pickModel('Large'),
              epochs: clampEp(550000),
              crop_size: cropWide,
              grid: 1,
              loss: { cr: 0.25, gw: 0.12, ha: cOut === 1 ? 0 : 0.5 },
              stage: [15, 25, 60],
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'Med_SparseKF',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_bf_4',
              model_size: pickModel('Medium'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.1, ha: cOut === 1 ? 0 : 0.5 },
              stage: [15, 25, 60],
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'CrispOptical_G18',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_bf_5',
              model_size: pickModel('Large'),
              epochs: clampEp(550000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.18, ha: cOut === 1 ? 0 : 0.5 },
              stage: [15, 25, 60],
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'SDR_NoHDRDamp',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_bf_6',
              model_size: pickModel('Large'),
              epochs: clampEp(500000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.1, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'Small_2FrameSafe',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_bf_7',
              model_size: pickModel('Small'),
              epochs: clampEp(450000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'NukeBase',
              role: 'control',
              reasonKey: 'lab.comboReason_nuke',
              nukeNative: true,
              model_size: pickModel('Large'),
              epochs: clampEp(500000),
              crop_size: clampCrop(256),
            },
          ];
        } else if (effType === 'degrain_clean') {
          specs = [
            {
              code: 'Anchor_Degrain',
              role: 'anchor',
              reasonKey: 'lab.comboReason_dg_1',
              model_size: pickModel('Small'),
              epochs: clampEp(400000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'CleanPlate_600k',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_dg_2',
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.35 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'CrispStruct_G15',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_dg_3',
              model_size: pickModel('Small'),
              epochs: clampEp(450000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.15, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'FlipReg_Small',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_dg_4',
              model_size: pickModel('Small'),
              epochs: clampEp(400000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'Med_CleanPrep',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_dg_5',
              model_size: pickModel('Medium'),
              epochs: clampEp(400000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'PureGrid1_MSE',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_dg_6',
              model_size: pickModel('Small'),
              epochs: clampEp(450000),
              crop_size: cropTight,
              grid: 1,
              loss: null,
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'WideCtx_Degrain',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_dg_7',
              model_size: pickModel('Small'),
              epochs: clampEp(400000),
              crop_size: cropWide,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: cOut === 1 ? 0 : 0.5 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'NukeBase',
              role: 'control',
              reasonKey: 'lab.comboReason_nuke',
              nukeNative: true,
              model_size: pickModel('Small'),
              epochs: clampEp(400000),
              crop_size: clampCrop(256),
            },
          ];
        } else {
          // roto_matte
          specs = [
            {
              code: 'Anchor_Roto',
              role: 'anchor',
              reasonKey: 'lab.comboReason_rm_1',
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'DeepMatte_900k',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_rm_2',
              model_size: pickModel('Small'),
              epochs: clampEp(900000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'SharpEdge_G12',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_rm_3',
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.3, gw: 0.12, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'PureGrid1_Alpha',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_rm_4',
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: cropTight,
              grid: 1,
              loss: null,
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'SymFlip_Roto',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_rm_5',
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: { flip: 0.5, ev: 0 },
            },
            {
              code: 'WideCtx_Small',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_rm_6',
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: cropWide,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'Med_DenseKF',
              role: 'relaxed',
              reasonKey: 'lab.comboReason_rm_7',
              model_size: pickModel('Medium'),
              epochs: clampEp(500000),
              crop_size: cropMax >= 384 ? cropWide : cropTight,
              grid: 1,
              loss: { cr: 0.25, gw: 0.05, ha: 0 },
              stage: null,
              ema: 0.9995,
              aug: null,
            },
            {
              code: 'NukeBase',
              role: 'control',
              reasonKey: 'lab.comboReason_nuke',
              nukeNative: true,
              model_size: pickModel('Small'),
              epochs: clampEp(600000),
              crop_size: clampCrop(256),
            },
          ];
        }

        const count = scope === 'compact' ? 4 : scope === 'thorough' ? 8 : 6;
        const controlSpec = specs[specs.length - 1];
        const chosen = count >= specs.length ? specs : specs.slice(0, count - 1).concat([controlSpec]);

        const nFrames = Math.max(1, Number((ds && ds.frame_count) || 8));
        const reqBs = form.batch_size_mode === 'Auto' ? 8 : Math.max(1, Math.floor(Number(form.batch_size) || 8));
        const effBs = Math.min(nFrames, reqBs);
        const stopEp = labComboCfg.stopAfterOptimalOn
          ? Math.max(100, Math.floor(Number(labComboCfg.stopAfterOptimalEpochs) || 50000))
          : 0;
        const autoExt = !!(ds && ds.has_eval_sequence && labComboCfg.autoExtendOn);

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
            if (sp.ema) {
              advSnap.ema_lr.enabled = true;
              advSnap.ema_lr.ema_decay = sp.ema;
            }
            if (sp.aug && (sp.aug.flip > 0 || sp.aug.ev > 0)) {
              advSnap.augment.enabled = true;
              advSnap.augment.hflip_prob = sp.aug.flip;
              advSnap.augment.exposure_ev = cIn < 3 && cOut < 3 ? 0 : sp.aug.ev;
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
              auto_extend_optimal: autoExt,
              stop_after_optimal_epochs: stopEp,
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
          formSnap.auto_extend_optimal = autoExt;
          formSnap.stop_after_optimal_epochs = stopEp;
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
        const estPatienceParallelSec = labComboCfg.stopAfterOptimalOn
          ? Math.max(10, Math.round(maxBudgetParallelSec * 0.68))
          : maxBudgetParallelSec;

        return {
          effType,
          gpuSlots,
          candidates,
          maxBudgetParallelSec,
          estPatienceParallelSec,
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
        if (labComboCfg.stopAfterOptimalOn) {
          const stopVal = Math.floor(Number(labComboCfg.stopAfterOptimalEpochs || 0));
          if (!Number.isFinite(stopVal) || stopVal < 1) {
            notify(t('msg.errInvalidStopOptimalEpochs'), 'error');
            return;
          }
          labComboCfg.stopAfterOptimalEpochs = stopVal;
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
            if (topNav.value === 'studio' && studioMode.value === 'monitor') drawLossChart();
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
        helpCurveTab,
        helpCompareShotFilter,
        filteredShotDeepReports,
        helpDeepCurveGroup,
        helpDeepShowSeen,
        helpDeepCurveChart,
        helpLedgerShot,
        helpLedgerModel,
        helpLedgerSort,
        helpLedgerQuery,
        helpPinnedRunIds,
        isHelpRunPinned,
        toggleHelpPinRun,
        setHelpPinBase,
        clearHelpPins,
        resetHelpDemoPins,
        filteredHelpLedger,
        pinnedComparisonRows,
        helpGuide,
        helpTelVars,
        helpGuideLoading,
        helpLoc,
        helpHeroBarPct,
        helpTriadBarPct,
        helpTrajectoryChart,
        applyHelpRecipe,
        helpBench,
        openHelp,
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
        advEmaPresets,
        isAdvModuleEffective,
        advActiveCount,
        advAnyItemExpanded,
        advStageStepsPreview,
        advHdrPreview,
        advEmaPreview,
        advExposurePreview,
        toggleAdvItem,
        toggleAdvAllItems,
        toggleRawLatex,
        copyLatexSource,
        resetAdvToNukeNative,
        applyAdvPreset,
        onStageRatioInput,
        syncLossFamilyEnabled,
        onLossFamilyToggle,
        syncAugmentEnabled,
        onAugmentToggle,
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
        logPreRef,
        logAutoScroll,
        uploadModal,
        datasetPickerModal,
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
        evaluateTaskRecommendations,
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
        setLabComboPatiencePreset,
        onLabComboPatienceInput,
        labComboSuitePreview,
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
  }).mount('#app');
}

mountWildCatApp();
