import { createApp, ref, reactive, computed, onMounted, nextTick, watch } from '/static/vendor/vue.esm.js';

const AUTO_BATCH = { Small: 8, Medium: 8, Large: 8 }; // CopyCat 'Auto' batch = 8 (memory permitting)

createApp({
  setup() {
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

    // ───────── Machine Identity & Admin Auth State ─────────
    const user = ref(null);
    const buildLabel = ref('v2.2.0');
    const savedMachine = (localStorage.getItem('wc_machine_name') || '')
      .replace(/[^A-Za-z0-9._-]/g, '_')
      .replace(/^[._]+|[._]+$/g, '');
    const initMachine = /[A-Za-z0-9]{2,}/.test(savedMachine) ? savedMachine : '';
    const machineName = ref(initMachine);
    const machineNameInput = ref(initMachine);
    const apiKey = ref('');
    const adminToken = ref(sessionStorage.getItem('wc_admin_token') || '');
    const adminPwdInput = ref('');
    const adminAuthLoading = ref(false);
    const adminAuthError = ref('');
    const pwdForm = reactive({ old_password: '', new_password: '' });

    // ───────── Top Navigation & Studio Mode ─────────
    const topNav = ref('studio'); // 'studio' | 'assets' | 'api' | 'admin'
    const studioMode = ref('create'); // 'create' | 'monitor'
    const initCategory = ref('scratch'); // 'scratch' | 'pretrained' | 'checkpoint'
    const showAdvanced = ref(true);

    const queue = reactive({ gpus: [], running: [], queued: [], queued_count: 0, running_count: 0 });
    const tasks = ref([]);
    const assets = ref([]);
    const selectedDatasetId = ref(null);
    const checkpoints = ref([]);
    const selectedTaskId = ref(null);
    const extendTargetSteps = ref(2000);
    const extendTargetEpochs = ref(20000);
    const helpBench = ref(null);

    async function openHelp() {
      topNav.value = 'help';
      if (helpBench.value) return;
      try {
        const r = await fetch('/static/benchmarks.json', { cache: 'no-cache' });
        helpBench.value = await r.json();
      } catch (e) {
        helpBench.value = { gpus: [], rows: [], setup: '速度数据加载失败', updated: '' };
      }
    }

    function benchTotalSpeedup(r) {
      // equal-length stages: total time ∝ Σ 1/sps
      const t = (a) => a.reduce((s, v) => s + 1 / v, 0);
      return (t(r.nuke) / t(r.wildcat)).toFixed(1);
    }

    // ───────── Step 1-3 Create Task Form (Standard Baseline Defaults) ─────────
    const form = reactive({
      run_name: 'WildCat1',
      preferred_gpu: 'auto',
      source: 'dataset/source',
      target: 'dataset/target',
      model_size: 'Medium', // Default: Medium
      in_channels: 3,       // Native Default: 3
      out_channels: 3,      // Native Default: 3
      epochs: 10000,        // Native Default: 10000
      batch_size_mode: 'Auto', // Native Default: Auto (Small=8, Medium=8, Large=4)
      batch_size: 8,
      crop_size_mode: 'Auto',  // Native Default: Auto (256)
      crop_size: 256,
      multi_resolution: true,  // Native Default: True
      learning_rate: 0.0001,   // Native Default: 1e-4 (poly power=0.3)
      precision: 'adaptive', // CopyCat 'Adaptive' mixed precision
      seed: 0,
      working_colorspace: 'ACEScg', // OCIO Default Working Colorspace: ACEScg
      output_format: 'exr_half',    // Default Output File Format: OpenEXR 16-bit half-float
      initial_weights_mode: 'None', // 'None' | 'Checkpoint' | built-in preset name
      checkpoint_path: '',
      checkpoint_interval: 1000, // Native Default: 1000
      preview_interval: 100,     // Native Default: 100
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
    const batchSizeChoice = ref('Auto');
    const cropSizeChoice = ref('Auto');

    const pathStats = reactive({
      sourceCount: 0,
      sourceW: 0,
      sourceH: 0,
      sourceCh: 3,
      sourceThumb: null,
      targetCount: 0,
      targetW: 0,
      targetH: 0,
      targetCh: 3,
      targetThumb: null,
    });

    // ───────── Monitor, 3x3 Contact Sheet & Model Versions State ─────────
    const telemetry = reactive({
      steps: [],
      vals: [],
      tail: [],
      latest: {},
    });
    const viewerModes = [
      { id: 'preview', label: '训练预览' },
      { id: 'source', label: 'Source' },
      { id: 'target', label: 'Target' },
    ];
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
    });
    const chartCfg = reactive({ logScale: true, smooth: 0.6 });
    const lossCanvas = ref(null);
    const logPreRef = ref(null);
    const logAutoScroll = ref(true);

    // ───────── Paired Dataset Upload Modal State ─────────
    const uploadModal = reactive({
      open: false,
      name: '',
      inputColorspace: 'ACEScg',
      enableCheck: true,
      strategy: 'standard',
      maxErrorPct: 35,
      sourceFiles: [],
      targetFiles: [],
      errorMsg: '',
      uploading: false,
    });

    // ───────── Tools & Admin State ─────────
    const adminSettings = reactive({ allow_self_registration_bool: false, max_jobs_per_gpu: 1 });
    const adminUsers = ref([]);
    const adminTasks = ref([]);
    const newUser = reactive({ username: '', password: '', role: 'user', note: '' });

    // ───────── Computed Properties & Architecture Locking Rules ─────────
    const selectedDataset = computed(() => {
      return assets.value.find((a) => a.id === selectedDatasetId.value) || assets.value[0] || null;
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

    // Channels (in_channels -> out_channels) are LOCKED whenever any Pretrained or Checkpoint weight is selected
    const isChannelsLocked = computed(() => {
      return initCategory.value === 'checkpoint' || initCategory.value === 'pretrained';
    });

    const computedTotalSteps = computed(() => {
      const nFrames = Math.max(1, (selectedDataset.value && selectedDataset.value.frame_count) || pathStats.sourceCount || 1);
      const bs = Math.min(nFrames, Math.max(1, Number(form.batch_size) || 8)); // batch never exceeds frame count
      const ep = Math.max(1, Number(form.epochs) || 1);
      return Math.max(1, Math.ceil((ep * nFrames) / bs));
    });

    const initSummaryText = computed(() => {
      if (initCategory.value === 'scratch') return 'None';
      if (initCategory.value === 'pretrained') return form.initial_weights_mode;
      return form.checkpoint_path || 'Checkpoint';
    });

    const activeTask = computed(() => {
      return tasks.value.find((t) => t.id === selectedTaskId.value) || tasks.value[0] || null;
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

    // Keep Auto Batch Size synchronized whenever form.model_size changes (whether manually or locked by weights)
    watch(
      () => form.model_size,
      (newSize) => {
        if (form.batch_size_mode === 'Auto') {
          form.batch_size = AUTO_BATCH[newSize] || 8;
        }
      }
    );

    watch(
      activeTask,
      (t) => {
        if (t) {
          extendTargetSteps.value = Math.max((t.total_steps || 1000) + 1000, (t.last_step || 0) + 1000);
          const ep = Number((t.config && t.config.train && t.config.train.epochs) || 0);
          extendTargetEpochs.value = ep > 0 ? ep * 2 : 20000;
        }
      },
      { immediate: true }
    );

    // ───────── HTTP Helper ─────────
    async function apiFetch(url, options = {}) {
      const headers = Object.assign({}, options.headers || {});
      const safeMachine = (machineName.value || '').replace(/[^\x20-\x7E]/g, '').trim();
      if (safeMachine) {
        headers['X-Machine-Name'] = safeMachine;
      }
      if (adminToken.value) {
        headers['X-Admin-Token'] = adminToken.value;
      }
      if (apiKey.value) {
        headers['X-API-Key'] = apiKey.value;
      }
      if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
        headers['Content-Type'] = 'application/json';
      }
      const res = await fetch(url, { ...options, headers });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (res.status === 403 && url.includes('/admin')) {
          adminToken.value = '';
          sessionStorage.removeItem('wc_admin_token');
        }
        const errMsg = Array.isArray(data.detail)
          ? data.detail.map((x) => x.msg || JSON.stringify(x)).join('; ')
          : typeof data.detail === 'object' && data.detail !== null
          ? JSON.stringify(data.detail)
          : data.detail;
        throw new Error(errMsg || `请求失败 (${res.status})`);
      }
      return data;
    }

    // ───────── Machine Session & Admin Auth Actions ─────────
    async function restoreSession() {
      try {
        const d = await apiFetch('/api/v1/auth/me');
        user.value = d.user;
        if (d.version && d.version.build_label) {
          buildLabel.value = d.version.build_label;
        }
        machineName.value = d.user.username;
        machineNameInput.value = d.user.username;
        apiKey.value = d.user.api_key || '';
        localStorage.setItem('wc_machine_name', d.user.username);
        await initWorkspace();
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    async function applyMachineName() {
      const raw = (machineNameInput.value || '').trim();
      const clean = raw.replace(/[^A-Za-z0-9._-]/g, '_').replace(/^[._]+|[._]+$/g, '');
      if (raw && !/[A-Za-z0-9]{2,}/.test(clean)) {
        notify('机器名请使用至少 2 位英文字母或数字（可含下划线/连字符）', 'error');
        return;
      }
      machineName.value = clean;
      if (clean) {
        localStorage.setItem('wc_machine_name', clean);
      } else {
        localStorage.removeItem('wc_machine_name');
      }
      apiKey.value = '';
      selectedDatasetId.value = null;
      selectedTaskId.value = null;
      await restoreSession();
      notify(`已切换至机器空间: ${machineName.value}`, 'success');
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
        sessionStorage.setItem('wc_admin_token', d.admin_token);
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
      sessionStorage.removeItem('wc_admin_token');
      topNav.value = 'studio';
    }

    async function changeAdminPassword() {
      if (!pwdForm.old_password || !pwdForm.new_password) {
        notify('请输入原管理员密码和新密码', 'error');
        return;
      }
      try {
        const d = await apiFetch('/api/v1/admin/password', {
          method: 'PUT',
          body: JSON.stringify(pwdForm),
        });
        adminToken.value = d.admin_token;
        sessionStorage.setItem('wc_admin_token', d.admin_token);
        pwdForm.old_password = '';
        pwdForm.new_password = '';
        notify(d.message || '管理员密码已加密更新', 'success');
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    // ───────── Workspace Data Loading ─────────
    async function initWorkspace() {
      await Promise.all([
        fetchQueueAndTasks(),
        fetchAssets(),
        fetchCheckpoints(),
      ]);
      proposeNextTaskName();
      await inspectFormPaths();

      const runningTask = tasks.value.find((t) => t.status === 'running' || t.status === 'queued');
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

      if (window.location.pathname === '/admin') {
        topNav.value = 'admin';
        if (adminToken.value) {
          await loadAdminData();
        }
      }
    }

    async function fetchQueueAndTasks() {
      if (!user.value) return;
      try {
        const [qRes, tRes] = await Promise.all([
          apiFetch('/api/v1/queue'),
          apiFetch('/api/v1/tasks'),
        ]);
        Object.assign(queue, qRes.queue);
        tasks.value = tRes.tasks || [];
        if (tasks.value.length && (!selectedTaskId.value || !tasks.value.some((x) => x.id === selectedTaskId.value))) {
          selectedTaskId.value = tasks.value[0].id;
        }
      } catch (_) {}
    }

    async function fetchAssets() {
      if (!user.value) return;
      try {
        const d = await apiFetch('/api/v1/assets');
        assets.value = d.assets || [];
        if (assets.value.length && (!selectedDatasetId.value || !assets.value.some((a) => a.id === selectedDatasetId.value))) {
          selectedDatasetId.value = assets.value[0].id;
        }
        syncFormWithSelectedDataset();
      } catch (_) {}
    }

    function syncFormWithSelectedDataset() {
      const ds = selectedDataset.value;
      if (!ds) {
        form.source = '';
        form.target = '';
        pathStats.sourceCount = 0;
        pathStats.targetCount = 0;
        pathStats.sourceW = 0;
        pathStats.sourceH = 0;
        pathStats.targetW = 0;
        pathStats.targetH = 0;
        pathStats.sourceThumb = null;
        pathStats.targetThumb = null;
        return;
      }
      form.source = ds.source_path || ds.rel_path || '';
      form.target = ds.target_path || ds.rel_path || '';
      pathStats.sourceCount = ds.frame_count || 0;
      pathStats.targetCount = ds.frame_count || 0;
      pathStats.sourceW = ds.width || 0;
      pathStats.sourceH = ds.height || 0;
      pathStats.sourceCh = ds.in_channels || 3;
      pathStats.targetW = ds.width || 0;
      pathStats.targetH = ds.height || 0;
      pathStats.targetCh = ds.out_channels || 3;
      pathStats.sourceThumb = ds.source_thumb_url || null;
      pathStats.targetThumb = ds.target_thumb_url || null;
      if (!isChannelsLocked.value && ds.in_channels && ds.out_channels) {
        const ic = Number(ds.in_channels || 3);
        const oc = Number(ds.out_channels || 3);
        if (oc === 1) {
          syncLockedChannels(ic >= 4 ? 4 : 3, 1);
        } else if (ic >= 4 && oc >= 4) {
          syncLockedChannels(4, 4);
        } else {
          syncLockedChannels(3, 3);
        }
      }
    }

    function onSelectDataset() {
      syncFormWithSelectedDataset();
    }

    async function fetchCheckpoints() {
      if (!user.value) return;
      try {
        const d = await apiFetch('/api/v1/checkpoints');
        const list = (d.checkpoints || []).filter((c) => c.path);
        checkpoints.value = list.length
          ? list
          : [{ path: '', label: '暂无已保存的检查点 (.pt)', model_size: 'Medium', in_channels: 3, out_channels: 3 }];
      } catch (_) {}
    }

    function proposeNextTaskName() {
      const existing = new Set(tasks.value.map((t) => t.node_name));
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
      inspectFormPaths();
    }

    function selectTaskForMonitor(t) {
      selectedTaskId.value = t.id;
      topNav.value = 'studio';
      studioMode.value = 'monitor';
      viewer.selectedPair = null;
      viewer.playing = false;
      contactViewer.playing = false;
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

    function onChannelPresetChange() {
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
        inspectFormPaths();
      }
    }

    function selectInitCategory(cat) {
      initCategory.value = cat;
      if (cat === 'scratch') {
        form.initial_weights_mode = 'None';
        form.checkpoint_path = '';
        form.model_size = 'Medium';
        syncLockedChannels(3, 3);
      } else if (cat === 'checkpoint') {
        form.initial_weights_mode = 'Checkpoint';
        if (!form.checkpoint_path && checkpoints.value.length && checkpoints.value[0].path) {
          form.checkpoint_path = checkpoints.value[0].path;
        }
        onCheckpointSelected();
      }
    }

    // ───────── Form & Hyperparameter Logic ─────────
    async function inspectFormPaths() {
      if (!user.value) return;
      if (!form.source || !form.target) {
        pathStats.sourceCount = 0;
        pathStats.targetCount = 0;
        pathStats.sourceW = 0;
        pathStats.sourceH = 0;
        pathStats.targetW = 0;
        pathStats.targetH = 0;
        pathStats.sourceThumb = null;
        pathStats.targetThumb = null;
        return;
      }
      try {
        const [sRes, tRes] = await Promise.all([
          apiFetch(`/api/v1/assets/inspect?path=${encodeURIComponent(form.source || '')}`),
          apiFetch(`/api/v1/assets/inspect?path=${encodeURIComponent(form.target || '')}`),
        ]);
        pathStats.sourceCount = sRes.frame_count || 0;
        pathStats.sourceW = sRes.width || 0;
        pathStats.sourceH = sRes.height || 0;
        pathStats.sourceCh = sRes.channels || 3;
        pathStats.sourceThumb = sRes.thumb_url || null;

        pathStats.targetCount = tRes.frame_count || 0;
        pathStats.targetW = tRes.width || 0;
        pathStats.targetH = tRes.height || 0;
        pathStats.targetCh = tRes.channels || 3;
        pathStats.targetThumb = tRes.thumb_url || null;
      } catch (_) {}
    }

    function onPickAssetFor(field, relPath) {
      if (!relPath) return;
      form[field] = relPath;
      inspectFormPaths();
    }

    function onModelSizeChange() {
      if (form.batch_size_mode === 'Auto') {
        form.batch_size = AUTO_BATCH[form.model_size] || 8;
      }
    }

    function onBsModeChange() {
      if (batchSizeChoice.value === 'Auto') {
        form.batch_size_mode = 'Auto';
        form.batch_size = AUTO_BATCH[form.model_size] || 8;
      } else {
        form.batch_size_mode = 'Manual';
        form.batch_size = Number(batchSizeChoice.value || 8);
      }
    }

    function onCropModeChange() {
      if (cropSizeChoice.value === 'Auto') {
        form.crop_size_mode = 'Auto';
        form.crop_size = 256;
      } else {
        form.crop_size_mode = 'Manual';
        form.crop_size = Number(cropSizeChoice.value || 256);
      }
    }

    // One master switch for all acceleration options (all lossless)
    const ACCEL_KEYS = ['opt_gpu_resident', 'opt_cuda_graph', 'opt_fused_adam', 'opt_channels_last', 'opt_cudnn_benchmark', 'opt_async_io'];
    const accelAll = computed({
      get: () => ACCEL_KEYS.every((k) => !!form[k]),
      set: (v) => ACCEL_KEYS.forEach((k) => { form[k] = !!v; }),
    });
    const accelAny = computed(() => ACCEL_KEYS.some((k) => !!form[k]));

    function buildConfigPayload(resume = false) {
      let iw = 'None';
      if (initCategory.value === 'checkpoint') {
        iw = form.checkpoint_path || '';
      }
      return {
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
          batch_size: Number(form.batch_size || 8),
          crop_size_auto: form.crop_size_mode === 'Auto',
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
            cuda_graph: !!form.opt_cuda_graph,
            fused_adam: !!form.opt_fused_adam,
            channels_last: !!form.opt_channels_last,
            cudnn_benchmark: !!form.opt_cudnn_benchmark,
            async_io: !!form.opt_async_io,
            preview_gpu: form.preview_gpu || 'auto',
          },
          lossy: {
            precision: form.precision || 'adaptive',
          },
        },
        loss: { l1_weight: 0.0, l2_weight: 1.0 },
        io: {
          checkpoint_interval: Math.max(10, Number(form.checkpoint_interval || 1000)),
          preview_enabled: true,
          preview_interval: Math.max(10, Number(form.preview_interval || 100)),
          preview_size: 0,
          log_interval: Math.max(1, Number(form.log_interval || 10)),
        },
      };
    }

    // ───────── 1. Launch Brand-New Training Task (Starts at Step 0) ─────────
    async function submitNewTrainingTask() {
      if (selectedDataset.value) {
        form.source = selectedDataset.value.source_path || selectedDataset.value.rel_path || '';
        form.target = selectedDataset.value.target_path || selectedDataset.value.rel_path || '';
      }
      if (!isChannelsLocked.value) {
        onChannelPresetChange();
      }
      if (!selectedDataset.value || !form.source || !form.target) {
        notify('请先在第一步选择一个配对训练数据集（或点击右上角新建配对数据集）', 'error');
        return;
      }
      if (initCategory.value === 'checkpoint' && !form.checkpoint_path) {
        notify('请选择一个已有的模型检查点 (.pt)，或切换回「从零开始全新训练」', 'error');
        return;
      }
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
        await fetchQueueAndTasks();
        await fetchCheckpoints();
        selectedTaskId.value = res.task.id;
        studioMode.value = 'monitor';
        proposeNextTaskName();
        if (res.task.status === 'running') {
          notify(`新训练任务 [${res.task.node_name}] 已在 GPU${res.task.assigned_gpu} 启动！`, 'success');
        } else {
          notify(`新任务 [${res.task.node_name}] 已加入队列排队，显卡空闲后将自动开始`, 'warning');
        }
        await fetchTelemetry(true);
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    // ───────── 2. Resume an Existing Interrupted Task (Continues from its saved breakpoint) ─────────
    async function resumeExistingTask(t) {
      if (!t) return;
      try {
        const curFormCfg = buildConfigPayload(true);
        const baseCfg = JSON.parse(JSON.stringify(t.config || curFormCfg));
        baseCfg.run_name = t.node_name;
        baseCfg.train = Object.assign({}, baseCfg.train || {}, {
          resume: true,
          precision: curFormCfg.train.precision,
        });
        baseCfg.optimizations = curFormCfg.optimizations;
        const res = await apiFetch('/api/v1/tasks', {
          method: 'POST',
          body: JSON.stringify({
            node_name: t.node_name,
            job_type: 'train',
            preferred_gpu: t.preferred_gpu || 'auto',
            config: baseCfg,
          }),
        });
        await fetchQueueAndTasks();
        selectedTaskId.value = res.task.id;
        studioMode.value = 'monitor';
        notify(`任务 [${t.node_name}] 已从第 ${t.last_step || 0} 步断点继续训练！`, 'success');
        await fetchTelemetry(true);
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    // ───────── 3. Extend Target Steps on an Existing Task (Same Model & Weights, More Steps) ─────────
    async function extendExistingTask(t) {
      if (!t) return;
      const curEpochs = Number((t.config && t.config.train && t.config.train.epochs) || 0);
      const newEpochs = Number(extendTargetEpochs.value || 0);
      const stepsPerEpoch = curEpochs > 0 && t.total_steps > 0 ? t.total_steps / curEpochs : 0;
      const newSteps = stepsPerEpoch > 0 ? Math.ceil(newEpochs * stepsPerEpoch) : 0;
      if (!newEpochs || (curEpochs && newEpochs <= curEpochs && newSteps <= (t.last_step || 0))) {
        notify(`目标 Epochs (${newEpochs}) 必须大于当前 Epochs (${curEpochs})`, 'error');
        return;
      }
      try {
        const baseCfg = JSON.parse(JSON.stringify(t.config || buildConfigPayload(true)));
        baseCfg.run_name = t.node_name;
        baseCfg.train = Object.assign({}, baseCfg.train || {}, { resume: true, epochs: newEpochs });
        delete baseCfg.train.total_steps; // recomputed as ceil(Epochs x frames / batch), same as CopyCat
        const res = await apiFetch('/api/v1/tasks', {
          method: 'POST',
          body: JSON.stringify({
            node_name: t.node_name,
            job_type: 'train',
            preferred_gpu: t.preferred_gpu || 'auto',
            config: baseCfg,
          }),
        });
        await fetchQueueAndTasks();
        selectedTaskId.value = res.task.id;
        studioMode.value = 'monitor';
        notify(`任务 [${t.node_name}] 继续训练至 ${newEpochs} Epochs (约 ${newSteps || '?'} Steps)`, 'success');
        await fetchTelemetry(true);
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    function retryFailedTask(t) {
      if (!t) return;
      const cfg = t.config || {};
      const paths = cfg.paths || {};
      const tr = cfg.train || {};
      const mo = cfg.model || {};
      form.run_name = t.node_name || form.run_name;
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
      } else {
        inspectFormPaths();
      }
      if (tr.epochs) form.epochs = Number(tr.epochs);
      if (mo.model_size) form.model_size = mo.model_size;
      topNav.value = 'studio';
      studioMode.value = 'create';
    }

    function taskEpoch(t) {
      const ep = Number((t && t.config && t.config.train && t.config.train.epochs) || 0);
      if (!t || !ep || !t.total_steps) return 0;
      return Math.min(ep, Math.floor(((t.last_step || 0) * ep) / t.total_steps));
    }

    // ───────── Monitor, Telemetry & Loss Chart ─────────
    function switchViewerMode(m) {
      viewer.mode = m;
      fetchTelemetry(true);
    }

    async function fetchTelemetry(resetToLatest = false) {
      if (!selectedTaskId.value || !user.value) return;
      try {
        const pairParam = viewer.selectedPair ? `&pair=${encodeURIComponent(viewer.selectedPair)}` : '';
        const d = await apiFetch(`/api/v1/tasks/${selectedTaskId.value}/telemetry?mode=${viewer.mode}${pairParam}`);
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
            notify(`任务 [${d.task.node_name}] 运行失败: ${lastLine || '请查看页面报错详情'}`, 'error');
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
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(300, rect.width) * dpr;
      canvas.height = Math.max(200, rect.height) * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const W = Math.max(300, rect.width);
      const H = Math.max(200, rect.height);
      ctx.clearRect(0, 0, W, H);

      const pts = (telemetry.steps || []).filter((s) => s.total != null && s.total > 0);
      if (!pts.length) {
        ctx.fillStyle = '#64748b';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('等待训练步数写入 Loss 数据...', W / 2, H / 2);
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

      const tf = (v) => (chartCfg.logScale ? Math.log10(Math.max(1e-7, v)) : v);
      const tRaw = rawYs.map(tf);
      const tSm = smYs.map(tf);
      const vPts = (telemetry.vals || []).filter((v) => v.mse != null && v.mse > 0);
      const tVal = vPts.map((v) => tf(v.mse));

      const allY = [...tRaw, ...tSm, ...tVal];
      let minX = Math.min(...xs), maxX = Math.max(...xs);
      if (minX === maxX) maxX = minX + 1;
      let minY = Math.min(...allY), maxY = Math.max(...allY);
      if (minY === maxY) { minY -= 0.5; maxY += 0.5; }

      const toX = (x) => padL + ((x - minX) / (maxX - minX)) * plotW;
      const toY = (y) => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

      // Horizontal Y Grid lines (5 intervals)
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

      // Vertical X Grid lines & Step Ticks
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
      const totSt = Number((activeTask.value && activeTask.value.total_steps) || tCfg.total_steps || 0);
      if (tCfg.multi_resolution !== false && totSt >= 9) {
        const s1 = Math.floor((totSt + 2) / 3);
        const s2 = s1 * 2;
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
        await apiFetch(`/api/v1/tasks/${taskId}/stop`, { method: 'POST' });
        notify('已暂停该任务并保存当前训练进度断点', 'warning');
        await fetchQueueAndTasks();
        await fetchCheckpoints();
        if (topNav.value === 'studio' && studioMode.value === 'monitor') await fetchTelemetry(false);
        if (topNav.value === 'admin') await loadAdminData();
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    async function deleteTask(t) {
      if (!confirm(`确定要删除任务 [${t.node_name}] 及其磁盘上的权重与预览文件吗？`)) return;
      try {
        await apiFetch(`/api/v1/tasks/${t.id}`, { method: 'DELETE' });
        notify(`已删除任务: ${t.node_name}`, 'warning');
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
          const stLabel = step != null ? `Step ${step}` : '最新';
          notify(`正在准备 Nuke 标准命名模型 (${stLabel})...`, 'info');
          const q = step != null ? `?step=${encodeURIComponent(step)}` : '';
          const expRes = await apiFetch(`/api/v1/tasks/${taskId}/export-cat${q}`, { method: 'POST' });
          if (ext === 'pt' && expRes.pt_filename) dlFilename = expRes.pt_filename;
          await fetchTelemetry(false);
        }
        const stepParam = step != null ? `&step=${encodeURIComponent(step)}` : '';
        const url = `/api/v1/tasks/${taskId}/download/${encodeURIComponent(ext)}?api_key=${encodeURIComponent(apiKey.value)}${stepParam}`;
        triggerBrowserDownload(url, dlFilename);
        notify(`开始下载模型: ${dlFilename || ('.' + ext)}`, 'success');
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    async function downloadAllModels(taskId, ext = 'pt') {
      if (!taskModels.count && !(activeTask.value && activeTask.value.has_ckpt)) {
        notify('当前任务尚未保存任何模型版本', 'warning');
        return;
      }
      try {
        const zipName = taskModels.zip_filename || `${taskModels.nuke_prefix || 'Training'}.zip`;
        notify(`正在打包本任务全部模型 (${zipName})，即将开始下载...`, 'info');
        const url = `/api/v1/tasks/${taskId}/download-all/${encodeURIComponent(ext)}?api_key=${encodeURIComponent(apiKey.value)}`;
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
        notify('当前暂无日志内容可复制', 'warning');
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
        notify('已复制完整日志与报错堆栈到剪贴板', 'success');
      } catch (_) {
        notify('浏览器剪贴板受限，请直接在日志框中选中文字复制', 'warning');
      }
    }

    // ───────── Paired Dataset Upload & Validation Actions ─────────
    function openDatasetModal() {
      uploadModal.name = `dataset_${Math.floor(Date.now() / 1000) % 10000}`;
      uploadModal.inputColorspace = 'ACEScg';
      uploadModal.enableCheck = true;
      uploadModal.strategy = 'standard';
      uploadModal.maxErrorPct = 35;
      uploadModal.sourceFiles = [];
      uploadModal.targetFiles = [];
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
      else uploadModal.targetFiles = list;
    }

    async function doUploadPairedDataset() {
      uploadModal.errorMsg = '';
      if (!uploadModal.sourceFiles.length || !uploadModal.targetFiles.length) {
        uploadModal.errorMsg = '必须同时选择「1. 输入原片序列 (Source)」和「2. 目标效果图序列 (Target)」才能创建配对数据集！';
        return;
      }
      uploadModal.uploading = true;
      try {
        const fd = new FormData();
        fd.append('name', uploadModal.name || 'paired_dataset');
        fd.append('input_colorspace', uploadModal.inputColorspace || 'ACEScg');
        fd.append('working_colorspace', 'ACEScg');
        fd.append('enable_check', uploadModal.enableCheck ? 'true' : 'false');
        fd.append('strategy', uploadModal.strategy || 'standard');
        fd.append('max_error_pct', String(uploadModal.maxErrorPct || 35));
        for (const f of uploadModal.sourceFiles) {
          fd.append('source_files', f);
        }
        for (const f of uploadModal.targetFiles) {
          fd.append('target_files', f);
        }
        const res = await apiFetch('/api/v1/assets/upload-pair', {
          method: 'POST',
          body: fd,
        });
        await fetchAssets();
        selectedDatasetId.value = res.asset.id;
        syncFormWithSelectedDataset();
        uploadModal.open = false;
        notify(
          `✓ 数据集 [${res.asset.name}] 创建成功（${res.asset.frame_count} 对帧 · 平均像素差 ${(res.asset.avg_mae * 100).toFixed(2)}%）`,
          'success'
        );
      } catch (e) {
        uploadModal.errorMsg = e.message;
      } finally {
        uploadModal.uploading = false;
      }
    }

    async function importDemoAssets() {
      try {
        await apiFetch('/api/v1/assets/import-demos', { method: 'POST' });
        await fetchAssets();
        notify('已恢复官方配对示例数据集', 'success');
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    function useDatasetForTraining(a) {
      selectedDatasetId.value = a.id;
      syncFormWithSelectedDataset();
      topNav.value = 'studio';
      studioMode.value = 'create';
    }

    async function deleteAsset(a) {
      if (!confirm(`确定要删除配对数据集 [${a.name}] 吗？`)) return;
      try {
        await apiFetch(`/api/v1/assets/${a.id}`, { method: 'DELETE' });
        await fetchAssets();
        notify(`已删除数据集: ${a.name}`, 'warning');
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    // ───────── API Key & Converter Tools ─────────
    async function regenerateApiKey() {
      try {
        const d = await apiFetch('/api/v1/auth/regenerate-key', { method: 'POST' });
        apiKey.value = d.api_key;
        user.value.api_key = d.api_key;
        localStorage.setItem('wc_api_key', d.api_key);
        notify('已生成新的 API Key', 'success');
      } catch (e) {
        notify(e.message, 'error');
      }
    }


    // ───────── Admin Console Actions ─────────
    async function switchToAdmin() {
      topNav.value = 'admin';
      if (adminToken.value) {
        await loadAdminData();
      }
    }

    async function loadAdminData() {
      if (!adminToken.value) return;
      try {
        const [sRes, uRes, tRes] = await Promise.all([
          apiFetch('/api/v1/admin/settings'),
          apiFetch('/api/v1/admin/users'),
          apiFetch('/api/v1/tasks?all_users=true'),
        ]);
        adminSettings.max_jobs_per_gpu = Number(sRes.settings.max_jobs_per_gpu || 1);
        adminUsers.value = uRes.users || [];
        adminTasks.value = tRes.tasks || [];
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    async function saveAdminSettings() {
      try {
        await apiFetch('/api/v1/admin/settings', {
          method: 'PUT',
          body: JSON.stringify({
            max_jobs_per_gpu: Number(adminSettings.max_jobs_per_gpu || 1),
          }),
        });
        notify('系统设置已保存生效', 'success');
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    async function adminCreateUser() {
      try {
        await apiFetch('/api/v1/admin/users', {
          method: 'POST',
          body: JSON.stringify({
            username: newUser.username,
            note: newUser.note,
          }),
        });
        notify(`已登记机器: ${newUser.username}`, 'success');
        newUser.username = '';
        newUser.note = '';
        await loadAdminData();
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    async function switchToMachineFromAdmin(targetMachine) {
      machineNameInput.value = targetMachine;
      await applyMachineName();
      topNav.value = 'studio';
    }

    async function adminPromptEditNote(u) {
      const n = prompt(`请输入机器 [${u.username}] 的备注说明:`, u.note || '');
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
      if (!confirm(`确定要删除机器 [${u.username}] 及其所有任务与数据集吗？`)) return;
      try {
        await apiFetch(`/api/v1/admin/users/${u.id}`, { method: 'DELETE' });
        notify(`已删除机器: ${u.username}`, 'warning');
        await loadAdminData();
      } catch (e) {
        notify(e.message, 'error');
      }
    }

    // ───────── Formatting Helpers ─────────
    function statusBadgeText(t) {
      if (!t) return '待启动';
      if (t.status === 'running') return `● 训练中 (GPU${t.assigned_gpu ?? 0})`;
      if (t.status === 'queued') return `⏳ 排队 #${t.queue_position || 1}`;
      if (t.status === 'completed') return '✓ 已完成';
      if (t.status === 'stopped') return '⏸ 已暂停';
      if (t.status === 'cancelled') return '已取消';
      if (t.status === 'failed') return '✕ 失败';
      return '待启动';
    }

    function statusLabel(t) {
      if (!t) return '待启动';
      if (t.status === 'running') return `● 正在训练 (GPU${t.assigned_gpu ?? 0})`;
      if (t.status === 'queued') return `⏳ 排队等待显卡 (#${t.queue_position || 1})`;
      if (t.status === 'completed') return '✓ 训练已完成';
      if (t.status === 'stopped') return `⏸ 已暂停 (停在第 ${t.last_step || 0} 步)`;
      if (t.status === 'cancelled') return '已取消';
      if (t.status === 'failed') return '✕ 运行失败';
      return '待启动';
    }

    function progressPct(t) {
      if (!t || !t.total_steps) return '0.0';
      return Math.min(100, ((t.last_step || 0) / t.total_steps) * 100).toFixed(1);
    }

    // ───────── Timers & Lifecycle ─────────
    watch([topNav, studioMode], ([nNav, nMode]) => {
      if (nNav === 'studio' && nMode === 'monitor') fetchTelemetry(true);
      if (nNav === 'admin' && adminToken.value) loadAdminData();
    });

    onMounted(() => {
      restoreSession();

      // Poll queue & telemetry every 1.8s
      setInterval(() => {
        if (!user.value) return;
        fetchQueueAndTasks();
        if (topNav.value === 'studio' && studioMode.value === 'monitor') {
          fetchTelemetry(false);
        }
      }, 1800);

      // Sequence & 3x3 Contact Sheet player loop (honoring individual FPS controls)
      let lastViewerTick = 0;
      let lastContactTick = 0;
      setInterval(() => {
        if (topNav.value !== 'studio' || studioMode.value !== 'monitor') return;
        const now = performance.now();
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
      }, 30);

      window.addEventListener('resize', () => {
        if (topNav.value === 'studio' && studioMode.value === 'monitor') drawLossChart();
      });
    });

    return {
      toasts,
      user,
      buildLabel,
      machineName,
      machineNameInput,
      applyMachineName,
      adminToken,
      adminPwdInput,
      adminAuthLoading,
      adminAuthError,
      pwdForm,
      doAdminLogin,
      logoutAdmin,
      changeAdminPassword,
      topNav,
      studioMode,
      initCategory,
      accelAll,
      accelAny,
      showAdvanced,
      queue,
      tasks,
      assets,
      selectedDatasetId,
      selectedDataset,
      checkpoints,
      selectedTaskId,
      extendTargetSteps,
      extendTargetEpochs,
      helpBench,
      openHelp,
      benchTotalSpeedup,
      taskEpoch,
      form,
      channelPreset,
      batchSizeChoice,
      cropSizeChoice,
      pathStats,
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
      adminSettings,
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
      inspectFormPaths,
      onModelSizeChange,
      onBsModeChange,
      onCropModeChange,
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
      exportAndDownloadCat: downloadModel,
      copyTrainLog,
      openDatasetModal,
      onUploadStrategyChange,
      onPairFilesSelected,
      doUploadPairedDataset,
      importDemoAssets,
      useDatasetForTraining,
      deleteAsset,
      regenerateApiKey,
      switchToAdmin,
      loadAdminData,
      saveAdminSettings,
      adminCreateUser,
      switchToMachineFromAdmin,
      adminPromptEditNote,
      adminToggleUserActive,
      adminDeleteUser,
      statusBadgeText,
      statusLabel,
      progressPct,
    };
  },
}).mount('#app');
