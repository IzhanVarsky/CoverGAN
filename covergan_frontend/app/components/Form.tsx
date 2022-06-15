import {
  Button,
  Container,
  Grid,
  InputWrapper,
  Loader,
  NativeSelect,
  NumberInput,
  Radio,
  RadioGroup,
  Stack,
  Switch,
  Text,
  TextInput,
} from '@mantine/core';
import Shape from './Shape';
import { useForm } from '@mantine/form';
import { Dropzone } from '@mantine/dropzone';
import { useOutletContext } from "@remix-run/react";
import { useState } from 'react';
import { FileMusic } from 'tabler-icons-react';
import config from '../config.json';

const emotions = [
  'ANGER',
  'COMFORTABLE',
  'FEAR',
  'FUNNY',
  'HAPPY',
  'INSPIRATIONAL',
  'JOY',
  'LONELY',
  'NOSTALGIC',
  'PASSIONATE',
  'QUIET',
  'RELAXED',
  'ROMANTIC',
  'SADNESS',
  'SERIOUS',
  'SOULFUL',
  'SURPRISE',
  'SWEET',
  'WARY'
];

export default function Form() {
  const [selectedCover, setSelectedCover, covers, setCovers] = useOutletContext();
  const [isLoading, setIsLoading] = useState(false);
  const [showError, setShowError] = useState(false);
  const [artistName, setArtistName] = useState('');
  const [trackName, setTrackName] = useState('');

  const splitByLast = (text: string, SEP: string = ".") => {
    const index = text.lastIndexOf(SEP);
    return index < 0 ? [text] : [text.slice(0, index), text.slice(index + SEP.length)];
  }

  const updArtistName = (name) => {
    form.setFieldValue("track_artist", name);
    setArtistName(name);
  }
  const updTrackName = (name) => {
    form.setFieldValue("track_name", name);
    setTrackName(name);
  }

  const setArtistAndTrackNames = (audioFilename) => {
    const fname = splitByLast(audioFilename, ".");
    if (fname.length < 2) {
      updArtistName(audioFilename);
      updTrackName('');
      return
    }
    const SEP = " - ";
    const arr = splitByLast(fname[0], SEP);
    if (arr.length == 2) {
      updArtistName(arr[0]);
      updTrackName(arr[1]);
    } else {
      updArtistName(arr[0]);
      updTrackName('');
    }
  }

  const sendData = (data) => {
    setIsLoading(true);
    console.log('Send data to server:', data);
    const formData = new FormData()
    for (let key in data) {
      formData.append(key, data[key]);
    }
    $.ajax({
      url: `${config.host}/generate`,
      type: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      cache: false,
      success: (response) => {
        console.log('SUCC', response);
        setCovers(response.result);
        setIsLoading(false);
      },
      error: (e) => {
        console.log('ERR', e);
        setIsLoading(false);
      }
    });
  };

  const form = useForm({
    initialValues: {
      audio_file: undefined,
      track_artist: { artistName },
      track_name: { trackName },
      emotion: 'anger',
      gen_type: "1",
      use_captioner: "1",
      num_samples: 5,
      use_filters: false,
    },
  });

  return (
    <>
      <Shape style={{ height: '100%', width: '23rem' }}>
        <form onSubmit={form.onSubmit((data) => {
          if (form.values['audio_file']) {
            setShowError(false);
            sendData(data);
          } else {
            setShowError(true);
          }
        })}>
          <Stack justify="space-around">
            <Text>Select music file</Text>
            <Dropzone
              multiple={false}
              accept={["audio/*"]}
              onDrop={(files) => {
                form.setFieldValue('audio_file', files[0]);
                setArtistAndTrackNames(files[0].name);
                setShowError(false);
              }}
              style={{
                borderColor: showError ? '#ffa3a3' : '',
                backgroundColor: showError ? '#fff6f5' : ''
              }}
            >
              {() =>
                <Grid columns={8} align='center'>
                  <Grid.Col span={1}>
                    <FileMusic color={showError ? '#ff3b3b' : 'grey'} size={30}/>
                  </Grid.Col>
                  <Grid.Col span={7}>
                    {form.values['audio_file']
                      ? <Text
                        color='grey'
                        style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}
                      >{form.values['audio_file'].name}</Text>
                      : <Text color={showError ? 'red' : 'grey'}>Drag or click</Text>}
                  </Grid.Col>
                </Grid>
              }
            </Dropzone>
            <TextInput
              placeholder={"Artist name"}
              label="Artist name"
              required
              onChange={(event) => updArtistName(event.currentTarget.value)}
              value={artistName}
            />
            <TextInput
              placeholder={"Track name"}
              label="Track name"
              required
              onChange={(event) => updTrackName(event.currentTarget.value)}
              value={trackName}
            />
            <RadioGroup
              label="Generator type"
              required
              {...form.getInputProps('gen_type')}
            >
              <Radio value="1" label="1"/>
              <Radio value="2" label="2"/>
            </RadioGroup>
            <RadioGroup
              label="Captioner type"
              required
              {...form.getInputProps('use_captioner')}
            >
              <Radio value="1" label="1"/>
              <Radio value="2" label="2"/>
            </RadioGroup>
            <NumberInput
              placeholder="Number of covers"
              label="Number of covers"
              min={1}
              max={20}
              required
              {...form.getInputProps('num_samples')}
            />
            <InputWrapper label="Emotion">
              <NativeSelect
                data={emotions}
                required
                {...form.getInputProps('emotion')}
              />
            </InputWrapper>
            <Switch
              size="md"
              label="Use color filters"
              {...form.getInputProps('use_filters')}
            />

            {isLoading
              ? <Container><Loader size='xl'/></Container>
              : <Button type='submit' variant='gradient'>Generate</Button>}
          </Stack>
        </form>
      </Shape>
    </>
  )
}