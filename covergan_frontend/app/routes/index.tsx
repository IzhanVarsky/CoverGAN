import { AppShell, Stack } from '@mantine/core';
import Header from '~/components/Header';
import Form from '~/components/Form';
import Covers from '~/components/covers/Covers';
import React from 'react';

export default class Index extends React.Component {
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
        navbar={<Stack><Header/><Form/></Stack>}
        styles={{
          main: {
            padding: 0,
          },
        }}
      >
        <Covers/>
      </AppShell>
    );
  }
}
