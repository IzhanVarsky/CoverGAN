import { AppShell } from '@mantine/core';
import React from 'react';
import Editor from '~/components/Editor';

export default class Edit extends React.Component {
  state = {
    loading: true
  };

  componentDidMount() {
    this.setState({ loading: false });
  }

  render() {
    if (this.state.loading) {
      return null;
    }

    return (
      <AppShell
        styles={{
          main: {
            padding: 0,
          },
        }}
      >
        <Editor/>
      </AppShell>
    );
  }
}
