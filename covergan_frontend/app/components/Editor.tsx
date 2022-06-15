import {
  ActionIcon,
  Button,
  Center,
  ColorInput,
  Grid,
  Group,
  NumberInput,
  ScrollArea,
  Stack,
  Tabs,
  Text,
  Textarea,
} from '@mantine/core';
import Shape from '~/components/Shape';
import { Link, useOutletContext } from '@remix-run/react';
import {
  AdjustmentsAlt,
  ArrowBackUp,
  ArrowBigLeft,
  ArrowForwardUp,
  Braces,
  Download,
  FileText,
  LayoutBoardSplit,
  Palette,
  Refresh,
} from 'tabler-icons-react';
import { downloadPNGFromServer, downloadTextFile, extractColors, getJSON } from "~/download_utils";
import {
  addRectBefore,
  addShadowFilter,
  changeAllColors,
  changeColorByIndex,
  getColors,
  getSVGSize,
  prettifyXml,
  svgWithSize
} from '~/utils';
import SVG from './SVG';
import { useState } from 'react';
import { Dropzone } from '@mantine/dropzone';
import useHistoryState from '~/HistoryState';

const randomColor = () => {
  const rgba = [];
  for (let i = 0; i < 3; i++) {
    rgba.push(Math.floor(Math.random() * 255));
  }
  rgba.push(Math.random().toFixed(2));
  return `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3]})`
};

export default function Main() {
  const [selectedCover, setSelectedCover, covers, setCovers] = useOutletContext();
  const [isLoading, setIsLoading] = useState(false);
  const [state, setState, undo, redo, history, pointer] = useHistoryState(covers.length ? {
    svg: prettifyXml(covers[selectedCover].svg),
    colors: getColors(covers[selectedCover].svg),
  } : { svg: '', colors: [] });
  const [imageSizeToDownload, setImageSizeToDownload] = useState(getSVGSize(state.svg).w);

  const updateStatePrettified = (newState) => {
    setState({
      svg: prettifyXml(newState.svg),
      colors: newState.colors
    })
  }

  const tryUpdateStateWithPrettified = () => {
    let p = prettifyXml(state.svg);
    if (p.includes('parsererror')) {
      return
    }
    setState({
      svg: p,
      colors: state.colors
    })
  }

  const updateSVGWithColors = (svg) => {
    updateStatePrettified({
      svg,
      colors: getColors(svg)
    })
  }

  const updCoverNotPrettified = (svg) => {
    // For textarea only
    let colors;
    if (prettifyXml(svg).includes('parsererror')) {
      colors = state.colors;
    } else {
      colors = getColors(svg);
    }
    setState({
      svg,
      colors,
    })
  }

  const updWithNewColors = (newColors) => {
    const newSVG = changeAllColors(state.svg, newColors);
    const newColorsState = newColors.map((el, i) => ({ attr: state.colors[i].attr, value: el }));
    updateStatePrettified({
      svg: newSVG,
      colors: newColorsState
    })
  }

  const updateColorByIndex = (ind) => (newColor) => {
    updateStatePrettified({
      svg: changeColorByIndex(state.svg, ind, newColor),
      colors: state.colors.map((el, i) => i === ind ? { attr: el.attr, value: newColor } : el),
    });
  }

  const shuffleColors = () => {
    function shuffle(array) {
      for (let i = array.length - 1; i > 0; i--) {
        let j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
      }
    }

    const newColors = state.colors.map((el, i) => el.value);
    shuffle(newColors);
    updWithNewColors(newColors);
  }

  return (
    <>
      <Link to="/">
        <Button m='md' leftIcon={<ArrowBigLeft/>} style={{ margin: 5, marginLeft: 16 }}>
          Go back to Main
        </Button>
      </Link>
      <Shape>
        <Grid justify='space-around' align="center" columns={2}>
          <Grid.Col span={1}>
            <Center>
              <SVG svg={state.svg}/>
            </Center>
            <Center>
              <Button m='md'
                      color={pointer == 0 ? 'gray' : ''}
                      onClick={undo}
                      leftIcon={<ArrowBackUp/>}
              >
                Undo
              </Button>
              <Button m='md'
                      color={pointer + 1 == history.length ? 'gray' : ''}
                      onClick={redo}
                      leftIcon={<ArrowForwardUp/>}
              >
                Redo
              </Button>
            </Center>
          </Grid.Col>
          <Grid.Col span={1}>
            <Tabs grow>
              <Tabs.Tab label="Edit Options" icon={<AdjustmentsAlt size={14}/>}>
                <Stack style={{ height: '70vh' }}>
                  <ScrollArea>
                    {state.colors.map((color, index) =>
                      <Center key={index}>
                        <Text>{color.attr}:</Text>
                        <ColorInput
                          style={{ margin: '10px', width: '50%' }}
                          value={color.value}
                          format='rgba'
                          onChange={updateColorByIndex(index)}
                          rightSection={
                            <ActionIcon onClick={() => updateColorByIndex(index)(randomColor())}>
                              <Refresh size={16}/>
                            </ActionIcon>
                          }
                        />
                      </Center>
                    )}
                  </ScrollArea>
                  <Button
                    style={{ minHeight: '5vh' }}
                    onClick={shuffleColors}
                  >
                    Shuffle colors
                  </Button>
                  <Dropzone
                    multiple={false}
                    accept={["image/*"]}
                    loading={isLoading}
                    onDrop={(files) => {
                      setIsLoading(true)
                      extractColors(files[0], state.colors.length, newColors => {
                        updWithNewColors(newColors);
                        setIsLoading(false)
                      }, () => setIsLoading(false));
                    }}>
                    {() =>
                      <Group style={{ pointerEvents: 'none' }}>
                        <Palette color='grey'/>
                        <Text color='grey'>Drop image to style transfer</Text>
                      </Group>
                    }
                  </Dropzone>
                  <Grid justify={'center'} align={'center'}>
                    <Grid.Col span={4}>
                      <Center>
                        <Button
                          onClick={() => updateSVGWithColors(addShadowFilter(state.svg))}>
                          Add shadow filter
                        </Button>
                      </Center>
                    </Grid.Col>
                    <Grid.Col span={4}>
                      <Center>
                        <Button
                          onClick={() => updateSVGWithColors(addRectBefore(state.svg))}>
                          Add color filter
                        </Button>
                      </Center>
                    </Grid.Col>
                  </Grid>
                </Stack>
              </Tabs.Tab>
              <Tabs.Tab label="Edit Raw SVG" icon={<FileText size={14}/>}>
                <Textarea
                  autoFocus={true}
                  placeholder='Write SVG . . .'
                  style={{ minHeight: '70vh' }}
                  minRows={10}
                  maxRows={21}
                  autosize
                  value={state.svg}
                  onChange={event => updCoverNotPrettified(event.currentTarget.value)}
                />
              </Tabs.Tab>
              <Tabs.Tab label="Download" icon={<Download size={14}/>}>
                <Stack style={{
                  padding: '0 25%',
                  justifyContent: 'flex-start', minHeight: '70vh'
                }}>
                  <NumberInput
                    defaultValue={imageSizeToDownload}
                    onChange={(val) => setImageSizeToDownload(val)}
                    min={0}
                    max={10000}
                    placeholder="Image size"
                    label="Image size"
                    required
                  />
                  <Button
                    leftIcon={<LayoutBoardSplit size={14}/>}
                    onClick={() => downloadTextFile(svgWithSize(state.svg, imageSizeToDownload), "edited.svg")}
                  >
                    Download SVG
                  </Button>
                  <Button
                    leftIcon={<Palette size={14}/>}
                    onClick={() => downloadPNGFromServer(svgWithSize(state.svg, imageSizeToDownload))}
                  >
                    Download PNG
                  </Button>
                  <Button
                    leftIcon={<Braces size={14}/>}
                    onClick={() => getJSON(svgWithSize(state.svg, imageSizeToDownload))}
                  >
                    Download JSON
                  </Button>
                </Stack>
              </Tabs.Tab>
              <Tabs.Tab disabled style={{ pointerEvents: 'none' }}
                        icon={
                          <Button component="span" variant="outline"
                                  style={{ pointerEvents: 'all' }}
                                  onClick={() => tryUpdateStateWithPrettified()}>
                            <Center style={{
                              height: "inherit",
                              display: "flex",
                              justifyContent: "center",
                              alignItems: "center"
                            }}>Prettify SVG</Center>
                          </Button>
                        }/>
            </Tabs>
          </Grid.Col>
        </Grid>
      </Shape>
    </>)
}
