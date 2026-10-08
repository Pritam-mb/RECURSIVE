import { Component, Fragment } from 'react';

/**
 * Contains a render/effect crash to one panel: shows a small
 * "panel error — retry" card instead of unmounting the whole app.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, key: 0 };
    this.retry = this.retry.bind(this);
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(`[ErrorBoundary:${this.props.name || 'panel'}]`, error, info?.componentStack);
  }

  retry() {
    this.props.onReset?.();
    this.setState((s) => ({ error: null, key: s.key + 1 }));
  }

  render() {
    const { error, key } = this.state;
    if (error) {
      return (
        <div
          role="alert"
          style={{
            position: this.props.overlay ? 'absolute' : 'relative',
            ...(this.props.overlay ? { left: 8, bottom: 8, zIndex: 30 } : { margin: 8 }),
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '6px 10px',
            background: 'var(--c-panel)',
            border: '1px solid rgba(255, 90, 79, 0.45)',
            borderRadius: 'var(--radius)',
            fontSize: 'var(--fs-small)',
            color: 'var(--c-text)',
          }}
        >
          <span className="ui-label" style={{ color: 'var(--c-warning)' }}>
            {this.props.name ? `${this.props.name} error` : 'Panel error'}
          </span>
          <button type="button" className="ui-btn" style={{ height: 22 }} onClick={this.retry}>Retry</button>
        </div>
      );
    }
    return <Fragment key={key}>{this.props.children}</Fragment>;
  }
}
