# CoverGAN frontend server
This application is a React framework based on [Remix](https://remix.run/docs).

### Installing dependencies
Install [NodeJS](https://nodejs.org/en/download/).

Run: `npm install`.

### Development

From your terminal:

```sh
npm run dev
```

This starts the app in development mode, rebuilding assets on file changes.

### Local Deployment

First, build the app for production:

```sh
npm run build
```

Then run the app in production mode:

```sh
npm start
```

Now you'll need to pick a host to deploy it to.

### Docker deployment
Specify the port of the app in [Dockerfile](Dockerfile) and in [running script](./docker_run_covergan_front.sh). By default the port is set to `5001`. 

Specify the host of backend service in [config.json](./app/config.json).

Build the docker image:
```sh
./docker_run_covergan_front.sh
```

Run the container:
```sh
./docker_build_covergan_front.sh
```