/**
 * Small reusable UI components: GlassCard, Button, Field, Modal, Tabs, Badge,
 * Toasts, Icon and Skeleton.
 * Styling lives in /static/css/ui.css and reads only tokens.css variables.
 */
(function () {
  const { h } = Vue;

  const pascal = (name) => String(name || '').replace(/(^|-)(\w)/g, (_, __, c) => c.toUpperCase());

  function renderNode(node) {
    if (!Array.isArray(node)) return null;
    const [tag, attrs, children] = node;
    return h(tag, attrs || {}, (children || []).map(renderNode));
  }

  const Icon = {
    name: 'UiIcon',
    props: { name: { type: String, required: true }, size: { type: Number, default: 18 } },
    render() {
      const lib = window.lucide && window.lucide.icons;
      const def = lib && lib[pascal(this.name)];
      if (!def) return h('span', { class: 'icon', 'aria-hidden': 'true' });
      const [, attrs, children] = def;
      return h(
        'svg',
        { ...attrs, width: this.size, height: this.size, class: 'icon', 'aria-hidden': 'true', focusable: 'false' },
        (children || []).map(renderNode)
      );
    },
  };

  const Button = {
    name: 'UiButton',
    components: { UiIcon: Icon },
    props: {
      variant: { type: String, default: 'default' }, // default | primary | ghost
      size: { type: String, default: 'md' },
      icon: { type: String, default: '' },
      loading: Boolean,
      disabled: Boolean,
      iconOnly: Boolean,
    },
    template: `<button type="button" class="btn" :class="['btn-' + variant, size === 'sm' ? 'btn-sm' : '', iconOnly ? 'btn-icon' : '']"
      :disabled="disabled || loading" :aria-busy="loading ? 'true' : 'false'">
      <ui-icon v-if="loading" name="loader-circle" class="spin" />
      <ui-icon v-else-if="icon" :name="icon" />
      <span v-if="$slots.default && !iconOnly" class="btn-label"><slot /></span>
    </button>`,
  };

  const Badge = {
    name: 'UiBadge',
    props: { tone: { type: String, default: 'muted' }, dot: Boolean },
    template: `<span class="badge" :class="'tone-' + tone"><span v-if="dot" class="dot"></span><slot /></span>`,
  };

  const GlassCard = {
    name: 'GlassCard',
    props: { title: String, subtitle: String, tag: { type: String, default: 'section' } },
    template: `<component :is="tag" class="glass card">
      <header v-if="title || $slots.actions" class="card-head">
        <div><h3 v-if="title" class="card-title">{{ title }}</h3><p v-if="subtitle" class="card-sub">{{ subtitle }}</p></div>
        <div v-if="$slots.actions" class="card-actions"><slot name="actions" /></div>
      </header>
      <slot />
    </component>`,
  };

  const Field = {
    name: 'UiField',
    props: {
      modelValue: { type: [String, Number], default: '' },
      label: String,
      placeholder: String,
      type: { type: String, default: 'text' },
      select: Boolean,
      id: String,
    },
    emits: ['update:modelValue'],
    data() { return { uid: 'f' + Math.random().toString(36).slice(2, 8) }; },
    template: `<div class="field">
      <label v-if="label" class="field-label" :for="id || uid">{{ label }}</label>
      <select v-if="select" class="input" :id="id || uid" :value="modelValue" @change="$emit('update:modelValue', $event.target.value)"><slot /></select>
      <input v-else class="input" :id="id || uid" :type="type" :placeholder="placeholder" :value="modelValue" @input="$emit('update:modelValue', $event.target.value)" />
    </div>`,
  };

  const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

  const Modal = {
    name: 'UiModal',
    components: { UiButton: Button },
    props: { open: Boolean, title: String, wide: Boolean },
    emits: ['close'],
    data() { return { returnFocus: null }; },
    watch: {
      open(value) {
        if (value) {
          this.returnFocus = document.activeElement;
          this.$nextTick(() => { const el = this.$refs.dialog; if (el) el.focus(); });
        } else if (this.returnFocus && this.returnFocus.focus) {
          this.returnFocus.focus();
          this.returnFocus = null;
        }
      },
    },
    methods: {
      onKey(event) {
        if (event.key === 'Escape') { event.stopPropagation(); this.$emit('close'); return; }
        if (event.key !== 'Tab') return;
        const nodes = [...this.$refs.dialog.querySelectorAll(FOCUSABLE)].filter((n) => n.offsetParent !== null);
        if (!nodes.length) { event.preventDefault(); return; }
        const first = nodes[0], last = nodes[nodes.length - 1];
        if (event.shiftKey && (document.activeElement === first || document.activeElement === this.$refs.dialog)) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      },
    },
    template: `<transition name="modal">
      <div v-if="open" class="modal-backdrop" @mousedown.self="$emit('close')">
        <div ref="dialog" class="modal" :class="{ wide }" role="dialog" aria-modal="true" :aria-label="title" tabindex="-1" @keydown="onKey">
          <header class="modal-head">
            <h2 class="modal-title">{{ title }}</h2>
            <ui-button variant="ghost" icon="x" icon-only size="sm" aria-label="Close dialog" @click="$emit('close')"></ui-button>
          </header>
          <div class="modal-body"><slot /></div>
          <footer v-if="$slots.footer" class="modal-foot"><slot name="footer" /></footer>
        </div>
      </div>
    </transition>`,
  };

  const Tabs = {
    name: 'UiTabs',
    props: { modelValue: String, items: { type: Array, required: true }, label: String, idPrefix: { type: String, default: 'tab' } },
    emits: ['update:modelValue'],
    methods: {
      move(event, index) {
        const keys = { ArrowRight: 1, ArrowLeft: -1, Home: 'first', End: 'last' };
        if (!(event.key in keys)) return;
        event.preventDefault();
        const n = this.items.length;
        const next = keys[event.key] === 'first' ? 0 : keys[event.key] === 'last' ? n - 1 : (index + keys[event.key] + n) % n;
        this.$emit('update:modelValue', this.items[next].key);
        this.$nextTick(() => { const el = this.$el.querySelector('[data-tab="' + this.items[next].key + '"]'); if (el) el.focus(); });
      },
    },
    template: `<div class="tabs" role="tablist" :aria-label="label">
      <button v-for="(item, i) in items" :key="item.key" type="button" class="tab" role="tab" :data-tab="item.key"
        :id="idPrefix + '-' + item.key" :aria-selected="modelValue === item.key ? 'true' : 'false'"
        :aria-controls="idPrefix + '-panel-' + item.key" :tabindex="modelValue === item.key ? 0 : -1"
        @click="$emit('update:modelValue', item.key)" @keydown="move($event, i)">
        {{ item.label }}
      </button>
    </div>`,
  };

  const Toasts = {
    name: 'UiToasts',
    components: { UiIcon: Icon },
    props: { items: { type: Array, default: () => [] } },
    template: `<div class="toasts" aria-live="polite" aria-atomic="false">
      <transition-group name="toast">
        <div v-for="t in items" :key="t.id" class="toast" :class="'tone-' + t.tone" :role="t.tone === 'danger' ? 'alert' : 'status'">
          <ui-icon :name="t.tone === 'danger' ? 'circle-alert' : t.tone === 'warn' ? 'triangle-alert' : 'circle-check'" />
          <span>{{ t.message }}</span>
        </div>
      </transition-group>
    </div>`,
  };

  const Skeleton = {
    name: 'UiSkeleton',
    props: { width: { type: String, default: '100%' }, height: { type: String, default: '16px' } },
    template: `<span class="skeleton" :style="{ width, height }" aria-hidden="true"></span>`,
  };

  window.FPLUI = {
    register(app) {
      app.component('UiIcon', Icon);
      app.component('UiButton', Button);
      app.component('UiBadge', Badge);
      app.component('GlassCard', GlassCard);
      app.component('UiField', Field);
      app.component('UiModal', Modal);
      app.component('UiTabs', Tabs);
      app.component('UiToasts', Toasts);
      app.component('UiSkeleton', Skeleton);
    },
  };
})();
